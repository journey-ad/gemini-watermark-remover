import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import sharp from 'sharp';

import {
    buildVideoWatermarkPolarityProbe,
    classifyVideoWatermarkEvidenceSummary,
    classifyVideoWatermarkFramePolarity,
    computeVideoBackgroundNormalizedAlphaContrast,
    detectVideoWatermarkFromFramesAsync,
    detectVideoWatermarkFromFrames,
    getVideoAlphaMap,
    resolveVideoAlphaEdgeBoost,
    scoreVideoWatermarkFramePolarity,
    summarizeVideoWatermarkFrameEvidence
} from '../../src/video/videoWatermarkDetector.js';
import * as videoWatermarkDetectorModule from '../../src/video/videoWatermarkDetector.js';
import { resolveVideoWatermarkCandidates } from '../../src/video/videoWatermarkCatalog.js';

function createPatternImageData(width, height) {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const value = (x * 5 + y * 3) % 180 + 30;
            data[idx] = value;
            data[idx + 1] = (value + 20) % 220;
            data[idx + 2] = (value + 40) % 240;
            data[idx + 3] = 255;
        }
    }
    return { width, height, data };
}

function applyWhiteWatermark(imageData, alphaMap, position) {
    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const alpha = alphaMap[row * position.width + col];
            if (alpha <= 0) continue;
            const idx = ((position.y + row) * imageData.width + position.x + col) * 4;
            for (let channel = 0; channel < 3; channel++) {
                imageData.data[idx + channel] = Math.round(alpha * 255 + (1 - alpha) * imageData.data[idx + channel]);
            }
        }
    }
}

function applyDarkWatermark(imageData, alphaMap, position) {
    for (let row = 0; row < position.height; row++) {
        for (let col = 0; col < position.width; col++) {
            const alpha = alphaMap[row * position.width + col];
            if (alpha <= 0) continue;
            const idx = ((position.y + row) * imageData.width + position.x + col) * 4;
            for (let channel = 0; channel < 3; channel++) {
                imageData.data[idx + channel] = Math.round((1 - alpha) * imageData.data[idx + channel]);
            }
        }
    }
}

async function decodeFixtureImageData(filePath) {
    const { data, info } = await sharp(filePath)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
    return {
        width: info.width,
        height: info.height,
        data: new Uint8ClampedArray(data)
    };
}

test('getVideoAlphaMap should use a softer edge boost for inset video candidates', () => {
    const [standard, inset] = resolveVideoWatermarkCandidates(1920, 1080);
    const standardAlpha = getVideoAlphaMap(standard.size, { candidate: standard });
    const insetAlpha = getVideoAlphaMap(inset.size, { candidate: inset });
    const explicitStandardAlpha = getVideoAlphaMap(standard.size, {
        edgeBoost: resolveVideoAlphaEdgeBoost(standard)
    });
    const explicitInsetAlpha = getVideoAlphaMap(inset.size, {
        edgeBoost: resolveVideoAlphaEdgeBoost(inset)
    });

    assert.equal(resolveVideoAlphaEdgeBoost(standard), 0.045);
    assert.equal(resolveVideoAlphaEdgeBoost(inset), 0.035);
    assert.equal(resolveVideoAlphaEdgeBoost({
        id: 'expected-anchor',
        size: 72,
        marginRight: 144,
        marginBottom: 144
    }), 0.035);
    assert.deepEqual(standardAlpha, explicitStandardAlpha);
    assert.deepEqual(insetAlpha, explicitInsetAlpha);
    assert.notDeepEqual(standardAlpha, insetAlpha);
});

test('getVideoAlphaMap should support experimental low-alpha scaling', () => {
    const [, inset] = resolveVideoWatermarkCandidates(1920, 1080);
    const currentAlpha = getVideoAlphaMap(inset.size, { candidate: inset });
    const scaledAlpha = getVideoAlphaMap(inset.size, {
        candidate: inset,
        lowAlphaScale: 0.92,
        bodyAlphaScale: 1.06
    });
    const lowIndex = currentAlpha.findIndex((alpha) => alpha > 0.04 && alpha < 0.10);
    const highIndex = currentAlpha.findIndex((alpha) => alpha >= 0.18);

    assert.ok(lowIndex >= 0);
    assert.ok(highIndex >= 0);
    assert.ok(scaledAlpha[lowIndex] < currentAlpha[lowIndex]);
    assert.ok(scaledAlpha[highIndex] > currentAlpha[highIndex]);
});

test('getVideoAlphaMap should support experimental embedded alpha profile selection', () => {
    const defaultAlpha = getVideoAlphaMap(96, { edgeBoost: 0 });
    const legacyAlpha = getVideoAlphaMap(96, { alphaProfile: '96', edgeBoost: 0 });
    const defaultMax = Math.max(...defaultAlpha);
    const legacyMax = Math.max(...legacyAlpha);

    assert.notDeepEqual(defaultAlpha, legacyAlpha);
    assert.ok(legacyMax > defaultMax, { defaultMax, legacyMax });
});

test('detectVideoWatermarkFromFrames should use the V1 alpha family for projected 32px video marks', () => {
    const width = 848;
    const height = 478;
    const target = resolveVideoWatermarkCandidates(width, height)
        .find((candidate) => candidate.id === 'veo-1080p-inset');
    const v1Alpha = getVideoAlphaMap(target.size, {
        candidate: target,
        alphaProfile: 48
    });
    const frames = [];

    for (let i = 0; i < 4; i++) {
        const imageData = createPatternImageData(width, height);
        applyWhiteWatermark(imageData, v1Alpha, {
            x: target.x,
            y: target.y,
            width: target.size,
            height: target.size
        });
        frames.push({ timestamp: i / 24, imageData });
    }

    const result = detectVideoWatermarkFromFrames({
        frames,
        width,
        height,
        candidates: [target],
        minConfidence: 0.02
    });

    assert.equal(result.candidate.id, target.id);
    assert.deepEqual(result.alphaMap, v1Alpha);
});

test('detectVideoWatermarkFromFrames should auto-select a legacy alpha shape for relocated portrait frames', () => {
    const width = 720;
    const height = 1280;
    const candidates = resolveVideoWatermarkCandidates(width, height);
    const target = candidates.find((candidate) => candidate.id === 'veo-720x1280-portrait-relocated-48');
    const legacyAlpha = getVideoAlphaMap(target.size, {
        candidate: target,
        alphaProfile: '96',
        edgeBoost: 0.12
    });
    const frames = [];

    for (let i = 0; i < 3; i++) {
        const imageData = createPatternImageData(width, height);
        applyWhiteWatermark(imageData, legacyAlpha, {
            x: target.x,
            y: target.y,
            width: target.size,
            height: target.size
        });
        frames.push({ timestamp: i / 24, imageData });
    }

    const result = detectVideoWatermarkFromFrames({
        frames,
        width,
        height,
        candidates: [target],
        minConfidence: 0.02
    });

    assert.equal(result.candidate.id, target.id);
    assert.equal(result.summary.alphaShape.accepted, true);
    assert.equal(result.summary.alphaShape.selected.profile, '96');
    assert.notEqual(result.summary.alphaShape.selected.name, 'default');
    assert.ok(result.summary.alphaShape.selected.edgeBoost <= 0.12);
    assert.deepEqual(result.alphaMap, getVideoAlphaMap(target.size, {
        candidate: target,
        alphaProfile: '96',
        edgeBoost: result.summary.alphaShape.selected.edgeBoost
    }));
});

test('detectVideoWatermarkFromFrames should auto-select legacy alpha on 20260619 relocated ROI fixtures', async () => {
    const fixtureDir = path.resolve('tests/fixtures/video-relocated-alpha/20260619');
    const frames = [];
    for (const [index, fileName] of ['roi-t1.png', 'roi-t5.png', 'roi-t9.png'].entries()) {
        frames.push({
            timestamp: index,
            imageData: await decodeFixtureImageData(path.join(fixtureDir, fileName))
        });
    }
    const candidate = {
        id: 'fixture-20260619-relocated-48',
        label: '20260619 relocated ROI fixture',
        x: 40,
        y: 40,
        size: 48,
        width: 48,
        height: 48,
        marginRight: 40,
        marginBottom: 40,
        sourceFamily: 'fixture',
        evidenceGate: 'required'
    };

    const result = detectVideoWatermarkFromFrames({
        frames,
        width: 128,
        height: 128,
        candidates: [candidate],
        minConfidence: 0.02
    });

    assert.equal(result.candidate.id, candidate.id);
    assert.equal(result.isConfident, true);
    assert.equal(result.summary.alphaShape.accepted, true);
    assert.equal(result.summary.alphaShape.selected.profile, '96');
    assert.notEqual(result.summary.alphaShape.selected.name, 'default');
    assert.ok(result.summary.alphaShape.baseline.meanConfidence > 0.75);
    assert.ok(result.summary.alphaShape.selected.meanConfidence < 0.36);
    assert.ok(
        result.summary.alphaShape.baseline.meanConfidence - result.summary.alphaShape.selected.meanConfidence > 0.4
    );
    assert.ok(result.alphaSeed.seedGain >= 1.2);
});

test('getVideoAlphaMap should support experimental local body scaling', () => {
    const [, inset] = resolveVideoWatermarkCandidates(1920, 1080);
    const currentAlpha = getVideoAlphaMap(inset.size, { candidate: inset });
    const scaledAlpha = getVideoAlphaMap(inset.size, {
        candidate: inset,
        localRegion: 'top-right',
        localBodyAlphaScale: 1.06
    });
    const size = inset.size;
    const center = (size - 1) / 2;
    const topRightIndex = currentAlpha.findIndex((alpha, index) => {
        const x = index % size;
        const y = Math.floor(index / size);
        return x >= center && y < center && alpha >= 0.12;
    });
    const bottomLeftIndex = currentAlpha.findIndex((alpha, index) => {
        const x = index % size;
        const y = Math.floor(index / size);
        return x < center && y >= center && alpha >= 0.12;
    });

    assert.ok(topRightIndex >= 0);
    assert.ok(bottomLeftIndex >= 0);
    assert.ok(scaledAlpha[topRightIndex] > currentAlpha[topRightIndex]);
    assert.equal(scaledAlpha[bottomLeftIndex], currentAlpha[bottomLeftIndex]);
});

test('detectVideoWatermarkFromFrames should vote for the candidate present across frames', () => {
    const width = 480;
    const height = 270;
    const candidates = resolveVideoWatermarkCandidates(width, height);
    const target = candidates[1];
    const alphaMap = getVideoAlphaMap(target.size, { candidate: target });
    const frames = [];

    for (let i = 0; i < 4; i++) {
        const imageData = createPatternImageData(width, height);
        applyWhiteWatermark(imageData, alphaMap, {
            x: target.x,
            y: target.y,
            width: target.size,
            height: target.size
        });
        frames.push({ timestamp: i / 24, imageData });
    }

    const result = detectVideoWatermarkFromFrames({
        frames,
        width,
        height,
        candidates,
        minConfidence: 0.02
    });

    assert.equal(result.candidate.id, target.id);
    assert.equal(result.position.x, target.x);
    assert.equal(result.position.y, target.y);
    assert.equal(result.isConfident, true);
});

test('detectVideoWatermarkFromFrames should preserve independent alternating watermark tracks', () => {
    const width = 192;
    const height = 128;
    const relocated = {
        id: 'alternating-relocated-32',
        label: 'alternating relocated 32',
        x: 12,
        y: 12,
        size: 32,
        width: 32,
        height: 32,
        sourceFamily: 'exact-size-exception',
        evidenceGate: 'required'
    };
    const standard = {
        id: 'alternating-standard-18',
        label: 'alternating standard 18',
        x: 158,
        y: 98,
        size: 18,
        width: 18,
        height: 18,
        sourceFamily: 'reference-projected',
        evidenceGate: 'standard'
    };
    const relocatedAlpha = getVideoAlphaMap(relocated.size, { candidate: relocated });
    const standardAlpha = getVideoAlphaMap(standard.size, { candidate: standard });
    const frames = [];

    for (let i = 0; i < 6; i++) {
        const imageData = createPatternImageData(width, height);
        const active = i < 3 ? relocated : standard;
        applyWhiteWatermark(
            imageData,
            active === relocated ? relocatedAlpha : standardAlpha,
            { x: active.x, y: active.y, width: active.size, height: active.size }
        );
        frames.push({ timestamp: i / 24, imageData });
    }

    const result = detectVideoWatermarkFromFrames({
        frames,
        width,
        height,
        candidates: [relocated, standard],
        minConfidence: 0.02
    });

    assert.deepEqual(
        result.detections?.map((detection) => detection.candidate.id).sort(),
        [relocated.id, standard.id].sort()
    );
    assert.equal(result.isConfident, true);
});

test('selectIndependentVideoWatermarkTracks should collapse overlapping catalog variants', () => {
    const summaries = [
        {
            candidate: { id: 'relocated-48', x: 576, y: 1136, size: 48, evidenceGate: 'required' },
            meanConfidence: 0.3909,
            maxConfidence: 0.9658,
            votes: 5
        },
        {
            candidate: { id: 'overlapping-inset-35', x: 583, y: 1149, size: 35, evidenceGate: 'required' },
            meanConfidence: 0.2198,
            maxConfidence: 0.4003,
            votes: 0
        },
        {
            candidate: { id: 'animated-compact-24', x: 648, y: 1208, size: 24, evidenceGate: 'required' },
            meanConfidence: 0.5341,
            maxConfidence: 0.9889,
            votes: 7
        }
    ];

    const selected = videoWatermarkDetectorModule.selectIndependentVideoWatermarkTracks?.(summaries, {
        minConfidence: 0.18
    });

    assert.deepEqual(selected?.map((summary) => summary.candidate.id), ['relocated-48', 'animated-compact-24']);
});

test('independent tracks should reject the zero-vote background candidate from issue 150', () => {
    const summaries = [
        {
            candidate: { id: 'relocated-48', x: 576, y: 1136, size: 48 },
            meanConfidence: 0.9404598428782002,
            frames: 12,
            votes: 12
        },
        {
            candidate: { id: 'animated-compact-24', x: 648, y: 1208, size: 24 },
            meanConfidence: 0.18923886692970018,
            frames: 12,
            votes: 0
        }
    ];
    assert.deepEqual(
        videoWatermarkDetectorModule.selectIndependentVideoWatermarkTracks(summaries),
        [summaries[0]]
    );
    // Preserve the primary best-effort result when sampling is inconclusive.
    const uncertain = { ...summaries[0], meanConfidence: 0.01, votes: 0 };
    assert.deepEqual(
        videoWatermarkDetectorModule.selectIndependentVideoWatermarkTracks([uncertain]),
        [uncertain]
    );
    // Issue 136's already-processed clip has inconsistent sampling. Keep its
    // established alternating path; forcing one track creates fresh dark dots.
    const mixedPrimary = { ...summaries[0], meanConfidence: 0.56466, votes: 8 };
    const fallback = { ...summaries[1], meanConfidence: 0.2141 };
    assert.deepEqual(
        videoWatermarkDetectorModule.selectIndependentVideoWatermarkTracks([mixedPrimary, fallback]),
        [mixedPrimary, fallback]
    );
});

test('selectVideoWatermarkDetectionForFrame should choose the active fallback track', () => {
    const width = 192;
    const height = 128;
    const relocated = {
        candidate: { id: 'relocated-32' },
        position: { x: 12, y: 12, width: 32, height: 32 },
        alphaMap: getVideoAlphaMap(32),
        meanConfidence: 0.4
    };
    const standard = {
        candidate: { id: 'standard-18' },
        position: { x: 158, y: 98, width: 18, height: 18 },
        alphaMap: getVideoAlphaMap(18),
        meanConfidence: 0.15
    };
    const data = new Uint8ClampedArray(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel++) {
        const idx = pixel * 4;
        data[idx] = 90;
        data[idx + 1] = 90;
        data[idx + 2] = 90;
        data[idx + 3] = 255;
    }
    const imageData = { width, height, data };
    applyWhiteWatermark(imageData, standard.alphaMap, standard.position);

    const selected = videoWatermarkDetectorModule.selectVideoWatermarkDetectionForFrame?.(
        imageData,
        { ...relocated, detections: [relocated, standard] }
    );

    assert.equal(selected?.detection.candidate.id, 'standard-18');
});

test('detectVideoWatermarkFromFramesAsync should preserve independent alternating watermark tracks', async () => {
    const width = 192;
    const height = 128;
    const relocated = {
        id: 'async-alternating-relocated-32',
        label: 'async alternating relocated 32',
        x: 12,
        y: 12,
        size: 32,
        width: 32,
        height: 32,
        sourceFamily: 'exact-size-exception',
        evidenceGate: 'required'
    };
    const standard = {
        id: 'async-alternating-standard-18',
        label: 'async alternating standard 18',
        x: 158,
        y: 98,
        size: 18,
        width: 18,
        height: 18,
        sourceFamily: 'reference-projected',
        evidenceGate: 'standard'
    };
    const relocatedAlpha = getVideoAlphaMap(relocated.size, { candidate: relocated });
    const standardAlpha = getVideoAlphaMap(standard.size, { candidate: standard });
    const frames = [];

    for (let i = 0; i < 6; i++) {
        const imageData = createPatternImageData(width, height);
        const active = i < 3 ? relocated : standard;
        applyWhiteWatermark(
            imageData,
            active === relocated ? relocatedAlpha : standardAlpha,
            { x: active.x, y: active.y, width: active.size, height: active.size }
        );
        frames.push({ timestamp: i / 24, imageData });
    }

    const result = await detectVideoWatermarkFromFramesAsync({
        frames,
        width,
        height,
        candidates: [relocated, standard],
        minConfidence: 0.02
    });

    assert.deepEqual(
        result.detections?.map((detection) => detection.candidate.id).sort(),
        [relocated.id, standard.id].sort()
    );
    assert.equal(result.isConfident, true);
});

test('detectVideoWatermarkFromFramesAsync yields while preserving selected detection', async () => {
    const width = 480;
    const height = 270;
    const candidates = resolveVideoWatermarkCandidates(width, height);
    const target = candidates[1];
    const alphaMap = getVideoAlphaMap(target.size, { candidate: target });
    const frames = [];

    for (let i = 0; i < 4; i++) {
        const imageData = createPatternImageData(width, height);
        applyWhiteWatermark(imageData, alphaMap, {
            x: target.x,
            y: target.y,
            width: target.size,
            height: target.size
        });
        frames.push({ timestamp: i / 24, imageData });
    }
    let yieldCount = 0;

    const result = await detectVideoWatermarkFromFramesAsync({
        frames,
        width,
        height,
        candidates,
        minConfidence: 0.02,
        yieldToMainThread: async () => {
            yieldCount++;
        }
    });

    assert.ok(yieldCount > 0);
    assert.equal(result.candidate.id, target.id);
    assert.equal(result.position.x, target.x);
    assert.equal(result.position.y, target.y);
    assert.equal(result.isConfident, true);
});

test('scoreVideoWatermarkFramePolarity should expose positive evidence for white video watermark', () => {
    const width = 480;
    const height = 270;
    const target = resolveVideoWatermarkCandidates(width, height)[0];
    const alphaMap = getVideoAlphaMap(target.size, { candidate: target });
    const imageData = createPatternImageData(width, height);
    const position = {
        x: target.x,
        y: target.y,
        width: target.size,
        height: target.size
    };

    applyWhiteWatermark(imageData, alphaMap, position);
    const score = scoreVideoWatermarkFramePolarity(imageData, position, alphaMap);

    assert.equal(score.bestPolarity, 'positive');
    assert.equal(score.shouldProcessCandidate, true);
    assert.equal(score.evidenceClass, 'positive-confident');
    assert.ok(score.polarityProbe.positiveScore > score.polarityProbe.negativeScore);
});

test('scoreVideoWatermarkFramePolarity should expose negative evidence for dark video watermark', () => {
    const width = 480;
    const height = 270;
    const target = resolveVideoWatermarkCandidates(width, height)[0];
    const alphaMap = getVideoAlphaMap(target.size, { candidate: target });
    const imageData = createPatternImageData(width, height);
    const position = {
        x: target.x,
        y: target.y,
        width: target.size,
        height: target.size
    };

    applyDarkWatermark(imageData, alphaMap, position);
    const score = scoreVideoWatermarkFramePolarity(imageData, position, alphaMap);

    assert.equal(score.bestPolarity, 'negative');
    assert.equal(score.shouldProcessCandidate, true);
    assert.equal(score.evidenceClass, 'negative-or-gray-polarity');
    assert.ok(score.polarityProbe.negativeScore > score.polarityProbe.positiveScore);
});

test('video polarity helpers should expose background-normalized evidence', () => {
    const imageData = {
        width: 2,
        height: 2,
        data: new Uint8ClampedArray([
            220, 220, 220, 255,
            50, 50, 50, 255,
            50, 50, 50, 255,
            50, 50, 50, 255
        ])
    };
    const alphaMap = new Float32Array([
        0.8, 0,
        0, 0
    ]);
    const background = computeVideoBackgroundNormalizedAlphaContrast(imageData, {
        x: 0,
        y: 0,
        width: 2,
        height: 2
    }, alphaMap);
    const probe = buildVideoWatermarkPolarityProbe(
        { spatial: 0.02, gradient: 0.01, confidence: 0.01 },
        background
    );
    const frame = classifyVideoWatermarkFramePolarity({
        spatial: -0.25,
        confidence: 0.01,
        ...buildVideoWatermarkPolarityProbe({ spatial: -0.25, gradient: 0.1, confidence: 0.01 })
    });

    assert.ok(background.alphaContrast > 0.6);
    assert.equal(probe.bestPolarity, 'gray');
    assert.equal(frame.bestPolarity, 'negative');
    assert.equal(frame.reason, 'negative-score-dominant');
});

test('video evidence summary should classify stable positive and negative-polarity batches', () => {
    const positiveSummary = summarizeVideoWatermarkFrameEvidence([
        { spatial: 0.5, gradient: 0.4, confidence: 0.43, bestPolarity: 'positive', shouldProcessCandidate: true },
        { spatial: 0.6, gradient: 0.5, confidence: 0.53, bestPolarity: 'positive', shouldProcessCandidate: true },
        { spatial: 0.4, gradient: 0.4, confidence: 0.4, bestPolarity: 'positive', shouldProcessCandidate: true },
        { spatial: 0.3, gradient: 0.3, confidence: 0.3, bestPolarity: 'positive', shouldProcessCandidate: true },
        { spatial: 0, gradient: 0, confidence: 0, bestPolarity: 'ambiguous', shouldProcessCandidate: false }
    ]);
    const negativeSummary = summarizeVideoWatermarkFrameEvidence([
        { spatial: -0.3, gradient: 0.04, confidence: 0.01, bestPolarity: 'negative', shouldProcessCandidate: true },
        { spatial: -0.25, gradient: 0.04, confidence: 0.01, bestPolarity: 'negative', shouldProcessCandidate: true },
        { spatial: -0.28, gradient: 0.04, confidence: 0.01, bestPolarity: 'negative', shouldProcessCandidate: true },
        { spatial: -0.22, gradient: 0.04, confidence: 0.01, bestPolarity: 'negative', shouldProcessCandidate: true },
        { spatial: -0.1, gradient: 0.01, confidence: 0.01, bestPolarity: 'negative', shouldProcessCandidate: false }
    ]);

    assert.equal(positiveSummary.positivePolarityFrames, 4);
    assert.equal(positiveSummary.processCandidateFrames, 4);
    assert.equal(
        classifyVideoWatermarkEvidenceSummary(positiveSummary).class,
        'positive-high-confidence'
    );
    assert.equal(negativeSummary.negativePolarityFrames, 5);
    assert.equal(
        classifyVideoWatermarkEvidenceSummary(negativeSummary).class,
        'negative-or-gray-polarity'
    );
});

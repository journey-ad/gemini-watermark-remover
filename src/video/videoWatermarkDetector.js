import { getEmbeddedAlphaMap } from '../core/embeddedAlphaMaps.js';
import {
    computeRegionGradientCorrelation,
    computeRegionSpatialCorrelation
} from '../core/adaptiveDetector.js';
import { resolveVideoWatermarkCandidates } from './videoWatermarkCatalog.js';
import {
    computeRectangularSpatialCorrelation,
    detectVeoTextWatermarkFromFramesAsync,
    detectVeoTextWatermarkFromFrames
} from './veoTextWatermarkDetector.js';

const DEFAULT_MIN_CONFIDENCE = 0.18;
const DEFAULT_ALPHA_SEED_GAIN = 1;
const VIDEO_ALPHA_PROFILE = '96-20260520';
const VIDEO_ALPHA_EDGE_BOOST = 0.045;
const VIDEO_INSET_ALPHA_EDGE_BOOST = 0.035;
const ALPHA_REFINEMENT_ROUNDS = 5;
const LOGO_VALUE = 255;
const AUTO_ALPHA_SHAPE_MIN_SIZE = 40;
const VIDEO_V1_ALPHA_MAX_SIZE = AUTO_ALPHA_SHAPE_MIN_SIZE - 1;
const AUTO_ALPHA_SHAPE_MIN_IMPROVEMENT = 0.04;
const AUTO_ALPHA_SHAPE_MIN_RELATIVE_IMPROVEMENT = 0.08;
const AUTO_ALPHA_SHAPE_MAX_EDGE_BOOST = 0.12;

function normalizeVideoAlphaProfile(profile) {
    if (profile === undefined || profile === null || profile === '') {
        return VIDEO_ALPHA_PROFILE;
    }

    const text = String(profile).trim();
    const numeric = Number(text);
    return Number.isInteger(numeric) && String(numeric) === text ? numeric : text;
}

function inferSquareAlphaSize(alphaMap, fallbackSize) {
    const size = Math.round(Math.sqrt(alphaMap.length));
    return size > 0 && size * size === alphaMap.length ? size : fallbackSize;
}

export function resizeAlphaMapArea(sourceAlpha, sourceSize, targetSize) {
    if (targetSize <= 0) return new Float32Array(0);
    if (sourceSize === targetSize) return new Float32Array(sourceAlpha);

    const out = new Float32Array(targetSize * targetSize);
    const scale = sourceSize / targetSize;

    for (let y = 0; y < targetSize; y++) {
        const yStart = y * scale;
        const yEnd = (y + 1) * scale;
        const y0 = Math.floor(yStart);
        const y1 = Math.ceil(yEnd);

        for (let x = 0; x < targetSize; x++) {
            const xStart = x * scale;
            const xEnd = (x + 1) * scale;
            const x0 = Math.floor(xStart);
            const x1 = Math.ceil(xEnd);

            let sum = 0;
            let areaSum = 0;
            for (let sy = y0; sy < y1; sy++) {
                if (sy < 0 || sy >= sourceSize) continue;
                const wy = Math.max(0, Math.min(yEnd, sy + 1) - Math.max(yStart, sy));
                for (let sx = x0; sx < x1; sx++) {
                    if (sx < 0 || sx >= sourceSize) continue;
                    const wx = Math.max(0, Math.min(xEnd, sx + 1) - Math.max(xStart, sx));
                    const area = wx * wy;
                    sum += sourceAlpha[sy * sourceSize + sx] * area;
                    areaSum += area;
                }
            }

            out[y * targetSize + x] = areaSum > 0 ? sum / areaSum : 0;
        }
    }

    return out;
}

export function resolveVideoAlphaEdgeBoost(candidate = null) {
    const size = Number(candidate?.size ?? candidate?.width ?? candidate?.height);
    const marginRight = Number(candidate?.marginRight);
    const marginBottom = Number(candidate?.marginBottom);
    const isInsetByGeometry = (
        Number.isFinite(size) &&
        size > 0 &&
        Number.isFinite(marginRight) &&
        Number.isFinite(marginBottom) &&
        marginRight / size >= 1.85 &&
        marginBottom / size >= 1.85
    );

    if (candidate?.id === 'veo-1080p-inset' || isInsetByGeometry) {
        return VIDEO_INSET_ALPHA_EDGE_BOOST;
    }
    return VIDEO_ALPHA_EDGE_BOOST;
}

function getVideoAlphaMap(size, options = {}) {
    const requestedProfile = options.profile ?? options.alphaProfile;
    const profile = requestedProfile === undefined || requestedProfile === null || requestedProfile === ''
        ? (size <= VIDEO_V1_ALPHA_MAX_SIZE ? 48 : VIDEO_ALPHA_PROFILE)
        : normalizeVideoAlphaProfile(requestedProfile);
    const alphaSource =
        getEmbeddedAlphaMap(profile) ||
        getEmbeddedAlphaMap(VIDEO_ALPHA_PROFILE) ||
        getEmbeddedAlphaMap(96);
    if (!alphaSource) {
        throw new Error('缺少 96px Gemini alpha map，无法生成视频水印模板');
    }
    const sourceSize = inferSquareAlphaSize(alphaSource, 96);
    const edgeBoost = Number.isFinite(options.edgeBoost)
        ? options.edgeBoost
        : resolveVideoAlphaEdgeBoost(options.candidate);
    const resized = size === sourceSize
        ? new Float32Array(alphaSource)
        : resizeAlphaMapArea(alphaSource, sourceSize, size);
    return applyVideoAlphaShapeOptions(enhanceVideoAlphaEdges(resized, size, edgeBoost), options);
}

function hasExplicitAlphaShapeOptions(options = {}) {
    return (
        options.profile !== undefined ||
        options.alphaProfile !== undefined ||
        options.edgeBoost !== undefined ||
        options.lowAlphaScale !== undefined ||
        options.bodyAlphaScale !== undefined ||
        options.localLowAlphaScale !== undefined ||
        options.localBodyAlphaScale !== undefined ||
        options.localRegion !== undefined
    );
}

function clampChannel(value) {
    if (value <= 0) return 0;
    if (value >= 255) return 255;
    return Math.round(value);
}

function createRestoredVideoRoi(imageData, position, alphaMap, alphaGain) {
    const width = position.width ?? position.size;
    const height = position.height ?? position.size;
    const out = new Uint8ClampedArray(width * height * 4);
    let deltaSum = 0;
    let changed = 0;
    let total = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const sourceIdx = ((position.y + y) * imageData.width + position.x + x) * 4;
            const targetIdx = (y * width + x) * 4;
            const rawAlpha = alphaMap[y * width + x] || 0;
            const alpha = rawAlpha > 0.025
                ? Math.min(rawAlpha * alphaGain, 0.99)
                : 0;

            if (alpha <= 0) {
                out[targetIdx] = imageData.data[sourceIdx];
                out[targetIdx + 1] = imageData.data[sourceIdx + 1];
                out[targetIdx + 2] = imageData.data[sourceIdx + 2];
                out[targetIdx + 3] = imageData.data[sourceIdx + 3] ?? 255;
            } else {
                const oneMinusAlpha = 1 - alpha;
                out[targetIdx] = clampChannel((imageData.data[sourceIdx] - alpha * LOGO_VALUE) / oneMinusAlpha);
                out[targetIdx + 1] = clampChannel((imageData.data[sourceIdx + 1] - alpha * LOGO_VALUE) / oneMinusAlpha);
                out[targetIdx + 2] = clampChannel((imageData.data[sourceIdx + 2] - alpha * LOGO_VALUE) / oneMinusAlpha);
                out[targetIdx + 3] = imageData.data[sourceIdx + 3] ?? 255;
            }

            for (let channel = 0; channel < 3; channel++) {
                const delta = Math.abs(out[targetIdx + channel] - imageData.data[sourceIdx + channel]);
                deltaSum += delta;
                if (delta > 1) changed++;
                total++;
            }
        }
    }

    return {
        imageData: {
            width,
            height,
            data: out
        },
        meanAbsDelta: total > 0 ? deltaSum / total : 0,
        changedRatio: total > 0 ? changed / total : 0
    };
}

function buildAutoAlphaShapeOptions(candidate, alphaMapOptions = {}) {
    const defaultEdgeBoost = resolveVideoAlphaEdgeBoost(candidate);
    return [
        {
            name: 'default',
            profile: normalizeVideoAlphaProfile(alphaMapOptions.profile ?? alphaMapOptions.alphaProfile),
            edgeBoost: Number.isFinite(alphaMapOptions.edgeBoost)
                ? alphaMapOptions.edgeBoost
                : defaultEdgeBoost,
            options: {}
        },
        {
            name: `96-edge${defaultEdgeBoost.toFixed(3)}`,
            profile: '96',
            edgeBoost: defaultEdgeBoost,
            options: {
                alphaProfile: '96',
                edgeBoost: defaultEdgeBoost
            }
        },
        ...[0.08, 0.10, AUTO_ALPHA_SHAPE_MAX_EDGE_BOOST].map((edgeBoost) => ({
            name: `96-edge${edgeBoost.toFixed(3)}`,
            profile: '96',
            edgeBoost,
            options: {
                alphaProfile: '96',
                edgeBoost
            }
        }))
    ];
}

function summarizeAutoAlphaShapeScores(scores) {
    if (!scores.length) {
        return {
            frames: 0,
            meanSpatial: 0,
            meanGradient: 0,
            meanConfidence: 1,
            maxConfidence: 1,
            meanAbsDelta: 0,
            changedRatio: 0
        };
    }

    const sum = scores.reduce((acc, score) => {
        acc.spatial += score.spatial;
        acc.gradient += score.gradient;
        acc.confidence += score.confidence;
        acc.maxConfidence = Math.max(acc.maxConfidence, score.confidence);
        acc.meanAbsDelta += score.meanAbsDelta;
        acc.changedRatio += score.changedRatio;
        return acc;
    }, {
        spatial: 0,
        gradient: 0,
        confidence: 0,
        maxConfidence: 0,
        meanAbsDelta: 0,
        changedRatio: 0
    });

    return {
        frames: scores.length,
        meanSpatial: sum.spatial / scores.length,
        meanGradient: sum.gradient / scores.length,
        meanConfidence: sum.confidence / scores.length,
        maxConfidence: sum.maxConfidence,
        meanAbsDelta: sum.meanAbsDelta / scores.length,
        changedRatio: sum.changedRatio / scores.length
    };
}

function rankAutoAlphaShapeEvaluation(evaluation, index) {
    const deltaPenalty = Math.max(0, evaluation.meanAbsDelta - 32) * 0.003;
    const coveragePenalty = Math.max(0, evaluation.changedRatio - 0.92) * 0.15;
    return evaluation.meanConfidence + deltaPenalty + coveragePenalty + index * 0.001;
}

function evaluateAutoAlphaShapeOption({
    frames,
    position,
    candidate,
    frameWinners,
    alphaMapOptions,
    option,
    scoreAlphaMap
}) {
    const alphaMap = getVideoAlphaMap(candidate.size, {
        ...alphaMapOptions,
        ...option.options,
        candidate
    });
    const alphaSeed = estimateAlphaSeedFromFrames(
        frames,
        position,
        alphaMap,
        frameWinners,
        candidate.id
    );
    const winnerByTimestamp = new Map(frameWinners.map((winner) => [winner.timestamp, winner]));
    const scores = [];

    for (const frame of frames) {
        const winner = winnerByTimestamp.get(frame.timestamp);
        if (winner && winner.candidateId !== candidate.id && winner.confidence > 0.08) continue;

        const restored = createRestoredVideoRoi(
            frame.imageData,
            position,
            alphaMap,
            alphaSeed.seedGain
        );
        const residual = scoreVideoWatermarkFrame(
            restored.imageData,
            { x: 0, y: 0, width: position.width, height: position.height },
            scoreAlphaMap
        );
        scores.push({
            ...residual,
            meanAbsDelta: restored.meanAbsDelta,
            changedRatio: restored.changedRatio
        });
    }

    return {
        ...option,
        alphaMap,
        alphaSeed,
        ...summarizeAutoAlphaShapeScores(scores)
    };
}

function shouldAutoSelectAlphaShape({ candidate, best, voteRatio, alphaMapOptions }) {
    if (alphaMapOptions?.autoAlphaProfile === false) return false;
    if (hasExplicitAlphaShapeOptions(alphaMapOptions)) return false;
    if (!candidate || !best) return false;
    if ((candidate.size ?? candidate.width ?? 0) < AUTO_ALPHA_SHAPE_MIN_SIZE) return false;
    if (voteRatio < 0.6) return false;
    return best.meanConfidence >= DEFAULT_MIN_CONFIDENCE;
}

function selectVideoAlphaShapeForDetection({
    frames,
    position,
    candidate,
    best,
    voteRatio,
    frameWinners,
    alphaMapOptions = {}
}) {
    const fallbackAlphaMap = getVideoAlphaMap(candidate.size, { ...alphaMapOptions, candidate });
    const fallbackAlphaSeed = estimateAlphaSeedFromFrames(
        frames,
        position,
        fallbackAlphaMap,
        frameWinners,
        candidate.id
    );

    if (!shouldAutoSelectAlphaShape({ candidate, best, voteRatio, alphaMapOptions })) {
        return {
            alphaMap: fallbackAlphaMap,
            alphaSeed: fallbackAlphaSeed,
            alphaShape: null
        };
    }

    const scoreAlphaMap = fallbackAlphaMap;
    const evaluations = buildAutoAlphaShapeOptions(candidate, alphaMapOptions)
        .map((option, index) => ({
            ...evaluateAutoAlphaShapeOption({
                frames,
                position,
                candidate,
                frameWinners,
                alphaMapOptions,
                option,
                scoreAlphaMap
            }),
            priorityIndex: index
        }))
        .sort((a, b) => {
            const aRank = rankAutoAlphaShapeEvaluation(a, a.priorityIndex);
            const bRank = rankAutoAlphaShapeEvaluation(b, b.priorityIndex);
            return aRank - bRank;
        });
    const baseline = evaluations.find((evaluation) => evaluation.name === 'default') || evaluations[0];
    const selected = evaluations[0] || baseline;
    const absoluteImprovement = baseline.meanConfidence - selected.meanConfidence;
    const relativeImprovement = baseline.meanConfidence > 0
        ? absoluteImprovement / baseline.meanConfidence
        : 0;
    const accepted =
        selected !== baseline &&
        absoluteImprovement >= AUTO_ALPHA_SHAPE_MIN_IMPROVEMENT &&
        relativeImprovement >= AUTO_ALPHA_SHAPE_MIN_RELATIVE_IMPROVEMENT;
    const finalSelection = accepted ? selected : baseline;

    return {
        alphaMap: finalSelection.alphaMap,
        alphaSeed: finalSelection.alphaSeed,
        alphaShape: {
            accepted,
            baseline: {
                name: baseline.name,
                profile: baseline.profile,
                edgeBoost: baseline.edgeBoost,
                meanConfidence: baseline.meanConfidence,
                meanAbsDelta: baseline.meanAbsDelta,
                changedRatio: baseline.changedRatio
            },
            selected: {
                name: finalSelection.name,
                profile: finalSelection.profile,
                edgeBoost: finalSelection.edgeBoost,
                meanConfidence: finalSelection.meanConfidence,
                meanAbsDelta: finalSelection.meanAbsDelta,
                changedRatio: finalSelection.changedRatio,
                alphaSeed: finalSelection.alphaSeed
            },
            candidates: evaluations.map((evaluation) => ({
                name: evaluation.name,
                profile: evaluation.profile,
                edgeBoost: evaluation.edgeBoost,
                meanConfidence: evaluation.meanConfidence,
                meanAbsDelta: evaluation.meanAbsDelta,
                changedRatio: evaluation.changedRatio
            }))
        }
    };
}

function applyVideoAlphaShapeOptions(alphaMap, options = {}) {
    const lowAlphaScale = Number.isFinite(options.lowAlphaScale)
        ? Math.max(0.5, Math.min(1.5, options.lowAlphaScale))
        : 1;
    const bodyAlphaScale = Number.isFinite(options.bodyAlphaScale)
        ? Math.max(0.5, Math.min(1.5, options.bodyAlphaScale))
        : 1;
    const localLowAlphaScale = Number.isFinite(options.localLowAlphaScale)
        ? Math.max(0.5, Math.min(1.5, options.localLowAlphaScale))
        : 1;
    const localBodyAlphaScale = Number.isFinite(options.localBodyAlphaScale)
        ? Math.max(0.5, Math.min(1.5, options.localBodyAlphaScale))
        : 1;
    const localRegion = typeof options.localRegion === 'string' ? options.localRegion : 'all';
    if (
        Math.abs(lowAlphaScale - 1) < 0.0001 &&
        Math.abs(bodyAlphaScale - 1) < 0.0001 &&
        Math.abs(localLowAlphaScale - 1) < 0.0001 &&
        Math.abs(localBodyAlphaScale - 1) < 0.0001
    ) {
        return alphaMap;
    }

    const size = Math.round(Math.sqrt(alphaMap.length));
    const out = new Float32Array(alphaMap.length);
    for (let i = 0; i < alphaMap.length; i++) {
        const alpha = alphaMap[i] || 0;
        const x = size > 0 ? i % size : 0;
        const y = size > 0 ? Math.floor(i / size) : 0;
        const inLocalRegion = matchesVideoAlphaLocalRegion(x, y, size, localRegion);
        if (alpha > 0 && alpha < 0.12) {
            const scale = lowAlphaScale * (inLocalRegion ? localLowAlphaScale : 1);
            out[i] = Math.min(0.99, alpha * scale);
        } else if (alpha >= 0.12) {
            const scale = bodyAlphaScale * (inLocalRegion ? localBodyAlphaScale : 1);
            out[i] = Math.min(0.99, alpha * scale);
        } else {
            out[i] = alpha;
        }
    }
    return out;
}

function matchesVideoAlphaLocalRegion(x, y, size, region) {
    if (!region || region === 'all') return true;
    const center = (size - 1) / 2;
    if (region === 'top') return y < center;
    if (region === 'bottom') return y >= center;
    if (region === 'left') return x < center;
    if (region === 'right') return x >= center;
    if (region === 'top-left') return y < center && x < center;
    if (region === 'top-right') return y < center && x >= center;
    if (region === 'bottom-left') return y >= center && x < center;
    if (region === 'bottom-right') return y >= center && x >= center;
    return true;
}

function enhanceVideoAlphaEdges(alphaMap, size, strength) {
    if (!Number.isFinite(strength) || strength <= 0 || size <= 2) {
        return new Float32Array(alphaMap);
    }

    const gradient = new Float32Array(alphaMap.length);
    let maxGradient = 0;

    for (let y = 1; y < size - 1; y++) {
        for (let x = 1; x < size - 1; x++) {
            const i = y * size + x;
            const gx =
                -alphaMap[i - size - 1] - 2 * alphaMap[i - 1] - alphaMap[i + size - 1] +
                alphaMap[i - size + 1] + 2 * alphaMap[i + 1] + alphaMap[i + size + 1];
            const gy =
                -alphaMap[i - size - 1] - 2 * alphaMap[i - size] - alphaMap[i - size + 1] +
                alphaMap[i + size - 1] + 2 * alphaMap[i + size] + alphaMap[i + size + 1];
            const value = Math.sqrt(gx * gx + gy * gy);
            gradient[i] = value;
            if (value > maxGradient) maxGradient = value;
        }
    }

    if (maxGradient <= 0) return new Float32Array(alphaMap);

    const out = new Float32Array(alphaMap.length);
    for (let i = 0; i < alphaMap.length; i++) {
        const edge = Math.sqrt(gradient[i] / maxGradient);
        out[i] = Math.min(0.99, alphaMap[i] + edge * strength);
    }
    return out;
}

function scoreCandidateOnFrame(imageData, candidate, alphaMapOptions = {}) {
    const alphaMap = getVideoAlphaMap(candidate.size, { ...alphaMapOptions, candidate });
    const region = {
        x: candidate.x,
        y: candidate.y,
        size: candidate.size
    };
    const spatial = computeRegionSpatialCorrelation({ imageData, alphaMap, region });
    const gradient = computeRegionGradientCorrelation({ imageData, alphaMap, region });
    const confidence = Math.max(0, spatial) * 0.35 + Math.max(0, gradient) * 0.65;

    return {
        candidate,
        alphaMap,
        spatial,
        gradient,
        confidence
    };
}

export function scoreVideoWatermarkFrame(imageData, position, alphaMap) {
    const width = position.width ?? position.size;
    const height = position.height ?? position.size;
    if (width !== height || alphaMap.length !== width * width) {
        const spatial = computeRectangularSpatialCorrelation({
            imageData,
            alphaMap,
            region: {
                x: position.x,
                y: position.y,
                width,
                height
            }
        });
        return {
            spatial,
            gradient: 0,
            confidence: Math.max(0, spatial)
        };
    }

    const region = {
        x: position.x,
        y: position.y,
        size: width
    };
    const spatial = computeRegionSpatialCorrelation({ imageData, alphaMap, region });
    const gradient = computeRegionGradientCorrelation({ imageData, alphaMap, region });
    const confidence = Math.max(0, spatial) * 0.35 + Math.max(0, gradient) * 0.65;

    return {
        spatial,
        gradient,
        confidence
    };
}

function normalizedLumaAt(data, idx) {
    return (0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2]) / 255;
}

export function computeVideoBackgroundNormalizedAlphaContrast(imageData, position, alphaMap, {
    highAlphaThreshold = 0.18,
    lowAlphaThreshold = 0.035
} = {}) {
    if (!imageData || !position || !alphaMap || alphaMap.length === 0) {
        return {
            alphaContrast: 0,
            normalizedAlphaContrast: 0,
            foregroundMean: null,
            backgroundMean: null,
            backgroundStdDev: 0
        };
    }

    const width = position.width ?? position.size;
    const height = position.height ?? position.size;
    let foregroundSum = 0;
    let foregroundWeight = 0;
    let backgroundSum = 0;
    let backgroundSq = 0;
    let backgroundWeight = 0;

    for (let y = 0; y < height; y++) {
        const imageY = position.y + y;
        if (imageY < 0 || imageY >= imageData.height) continue;
        for (let x = 0; x < width; x++) {
            const imageX = position.x + x;
            if (imageX < 0 || imageX >= imageData.width) continue;

            const alpha = alphaMap[y * width + x] || 0;
            const idx = (imageY * imageData.width + imageX) * 4;
            const luma = normalizedLumaAt(imageData.data, idx);

            if (alpha >= highAlphaThreshold) {
                foregroundSum += luma * alpha;
                foregroundWeight += alpha;
            } else if (alpha <= lowAlphaThreshold) {
                backgroundSum += luma;
                backgroundSq += luma * luma;
                backgroundWeight++;
            }
        }
    }

    const foregroundMean = foregroundWeight > 0 ? foregroundSum / foregroundWeight : null;
    const backgroundMean = backgroundWeight > 0 ? backgroundSum / backgroundWeight : null;
    const backgroundVariance = backgroundWeight > 0
        ? Math.max(0, backgroundSq / backgroundWeight - backgroundMean * backgroundMean)
        : 0;
    const backgroundStdDev = Math.sqrt(backgroundVariance);
    const alphaContrast = foregroundMean !== null && backgroundMean !== null
        ? foregroundMean - backgroundMean
        : 0;

    return {
        alphaContrast,
        normalizedAlphaContrast: alphaContrast / Math.max(0.015, backgroundStdDev),
        foregroundMean,
        backgroundMean,
        backgroundStdDev
    };
}

export function buildVideoWatermarkPolarityProbe(score, backgroundProbe = {}) {
    const spatial = Number.isFinite(score?.spatial) ? score.spatial : 0;
    const gradient = Number.isFinite(score?.gradient) ? score.gradient : 0;
    const confidence = Number.isFinite(score?.confidence) ? score.confidence : 0;
    const absSpatialScore = Math.abs(spatial);
    const positiveScore = Math.max(0, spatial) * 0.65 + Math.max(0, gradient) * 0.35;
    const negativeScore = Math.max(0, -spatial) * 0.65 + Math.max(0, gradient) * 0.35;
    const backgroundNormalizedScore = Math.abs(backgroundProbe.normalizedAlphaContrast || 0);
    const bestPolarity = positiveScore >= 0.14 && positiveScore >= negativeScore
        ? 'positive'
        : negativeScore >= 0.14 && negativeScore > positiveScore
            ? 'negative'
            : backgroundNormalizedScore >= 0.5
                ? 'gray'
                : 'ambiguous';
    const polarityMargin = Math.abs(positiveScore - negativeScore);
    const shouldProcessCandidate = (
        bestPolarity === 'positive' ||
        bestPolarity === 'negative' ||
        bestPolarity === 'gray'
    ) && Math.max(positiveScore, negativeScore, backgroundNormalizedScore * 0.2, confidence) >= 0.14;
    const reason = bestPolarity === 'positive'
        ? 'positive-template-evidence'
        : bestPolarity === 'negative'
            ? 'negative-template-evidence'
            : bestPolarity === 'gray'
                ? 'background-normalized-alpha-contrast'
                : 'insufficient-polarity-evidence';

    return {
        positiveScore,
        negativeScore,
        absSpatialScore,
        backgroundNormalizedScore,
        backgroundAlphaContrast: backgroundProbe.alphaContrast || 0,
        normalizedAlphaContrast: backgroundProbe.normalizedAlphaContrast || 0,
        bestPolarity,
        polarityMargin,
        shouldProcessCandidate,
        reason
    };
}

export function classifyVideoWatermarkFramePolarity(score) {
    const confidence = Number.isFinite(score?.confidence) ? score.confidence : 0;
    const spatial = Number.isFinite(score?.spatial) ? score.spatial : 0;
    const gradient = Number.isFinite(score?.gradient) ? score.gradient : 0;
    const positiveScore = Number.isFinite(score?.positiveScore)
        ? score.positiveScore
        : Math.max(0, spatial) * 0.65 + Math.max(0, gradient) * 0.35;
    const negativeScore = Number.isFinite(score?.negativeScore)
        ? score.negativeScore
        : Math.max(0, -spatial) * 0.65 + Math.max(0, gradient) * 0.35;
    const absSpatial = Number.isFinite(score?.absSpatialScore)
        ? score.absSpatialScore
        : Math.abs(spatial);
    const polarityMargin = Number.isFinite(score?.polarityMargin)
        ? score.polarityMargin
        : Math.abs(positiveScore - negativeScore);

    if (positiveScore >= 0.14 && spatial >= 0.14 && positiveScore >= negativeScore) {
        return {
            class: 'positive-confident',
            bestPolarity: 'positive',
            polarity: 'positive',
            polarityMargin,
            shouldProcessCandidate: true,
            reason: 'positive-score-confident',
            recommendedGateAction: 'process'
        };
    }
    if (negativeScore >= 0.14 && spatial <= -0.14 && absSpatial >= 0.18) {
        return {
            class: 'negative-or-gray-polarity',
            bestPolarity: 'negative',
            polarity: 'negative',
            polarityMargin,
            shouldProcessCandidate: true,
            reason: 'negative-score-dominant',
            recommendedGateAction: 'inspect-polarity'
        };
    }
    if (confidence >= 0.035) {
        return {
            class: 'weak-positive',
            bestPolarity: positiveScore >= negativeScore ? 'positive' : 'ambiguous',
            polarity: spatial >= 0 ? 'positive' : 'mixed',
            polarityMargin,
            shouldProcessCandidate: false,
            reason: 'weak-positive-evidence',
            recommendedGateAction: 'seed-or-review'
        };
    }
    return {
        class: 'low-evidence',
        bestPolarity: negativeScore > positiveScore ? 'negative' : 'ambiguous',
        polarity: spatial < -0.14 ? 'negative' : 'none',
        polarityMargin,
        shouldProcessCandidate: false,
        reason: 'below-frame-threshold',
        recommendedGateAction: 'skip-or-review'
    };
}

export function scoreVideoWatermarkFramePolarity(imageData, position, alphaMap) {
    const score = scoreVideoWatermarkFrame(imageData, position, alphaMap);
    const backgroundProbe = computeVideoBackgroundNormalizedAlphaContrast(imageData, position, alphaMap);
    const polarityProbe = buildVideoWatermarkPolarityProbe(score, backgroundProbe);
    const frameEvidence = classifyVideoWatermarkFramePolarity({ ...score, ...polarityProbe });

    return {
        ...score,
        evidenceClass: frameEvidence.class,
        bestPolarity: frameEvidence.bestPolarity,
        polarity: frameEvidence.polarity,
        polarityMargin: frameEvidence.polarityMargin,
        shouldProcessCandidate: frameEvidence.shouldProcessCandidate,
        reason: frameEvidence.reason,
        recommendedGateAction: frameEvidence.recommendedGateAction,
        polarityProbe
    };
}

export function summarizeVideoWatermarkFrameEvidence(frameScores) {
    const scores = Array.isArray(frameScores) ? frameScores : [];
    if (!scores.length) {
        return {
            frames: 0,
            meanSpatial: 0,
            meanAbsSpatial: 0,
            meanGradient: 0,
            meanConfidence: 0,
            maxConfidence: 0,
            maxAbsSpatial: 0,
            positiveSpatialFrames: 0,
            negativeSpatialFrames: 0,
            positivePolarityFrames: 0,
            negativePolarityFrames: 0,
            grayPolarityFrames: 0,
            processCandidateFrames: 0,
            confidentFrames: 0,
            weakFrames: 0,
            likelyAbsentFrames: 0
        };
    }

    const summary = {
        frames: scores.length,
        meanSpatial: 0,
        meanAbsSpatial: 0,
        meanGradient: 0,
        meanConfidence: 0,
        maxConfidence: 0,
        maxAbsSpatial: 0,
        positiveSpatialFrames: 0,
        negativeSpatialFrames: 0,
        positivePolarityFrames: 0,
        negativePolarityFrames: 0,
        grayPolarityFrames: 0,
        processCandidateFrames: 0,
        confidentFrames: 0,
        weakFrames: 0,
        likelyAbsentFrames: 0
    };

    for (const score of scores) {
        const confidence = Number.isFinite(score.confidence) ? score.confidence : 0;
        const spatial = Number.isFinite(score.spatial) ? score.spatial : 0;
        const absSpatial = Math.abs(spatial);
        summary.meanSpatial += spatial;
        summary.meanAbsSpatial += absSpatial;
        summary.meanGradient += Number.isFinite(score.gradient) ? score.gradient : 0;
        summary.meanConfidence += confidence;
        summary.maxConfidence = Math.max(summary.maxConfidence, confidence);
        summary.maxAbsSpatial = Math.max(summary.maxAbsSpatial, absSpatial);
        if (spatial >= 0.14) {
            summary.positiveSpatialFrames++;
        } else if (spatial <= -0.14) {
            summary.negativeSpatialFrames++;
        }
        if (score.bestPolarity === 'positive') summary.positivePolarityFrames++;
        if (score.bestPolarity === 'negative') summary.negativePolarityFrames++;
        if (score.bestPolarity === 'gray') summary.grayPolarityFrames++;
        if (score.shouldProcessCandidate === true) summary.processCandidateFrames++;
        if (confidence >= 0.14) {
            summary.confidentFrames++;
        } else if (confidence >= 0.035) {
            summary.weakFrames++;
        } else {
            summary.likelyAbsentFrames++;
        }
    }

    summary.meanSpatial /= scores.length;
    summary.meanAbsSpatial /= scores.length;
    summary.meanGradient /= scores.length;
    summary.meanConfidence /= scores.length;
    return summary;
}

export function classifyVideoWatermarkEvidenceSummary(summary) {
    const frames = Number.isFinite(summary?.frames) ? summary.frames : 0;
    if (frames <= 0) {
        return {
            class: 'unknown',
            shortLabel: 'unknown',
            label: 'unknown evidence',
            polarity: 'unknown',
            recommendedNextStep: 'collect-frames'
        };
    }

    const strongFrameCount = Math.max(2, Math.ceil(frames * 0.6));
    const halfFrameCount = Math.ceil(frames * 0.5);
    const confidentFrames = summary.confidentFrames || 0;
    const likelyAbsentFrames = summary.likelyAbsentFrames || 0;
    const negativeSpatialFrames = summary.negativeSpatialFrames || 0;
    const positiveSpatialFrames = summary.positiveSpatialFrames || 0;
    const meanAbsSpatial = summary.meanAbsSpatial || 0;
    const meanSpatial = summary.meanSpatial || 0;

    if (
        confidentFrames >= strongFrameCount &&
        positiveSpatialFrames >= strongFrameCount &&
        meanSpatial >= 0.14
    ) {
        return {
            class: 'positive-high-confidence',
            shortLabel: 'positive-high',
            label: 'positive high-confidence watermark',
            polarity: 'positive',
            recommendedNextStep: 'use-standard-detection-gate'
        };
    }

    if (
        negativeSpatialFrames >= strongFrameCount &&
        meanAbsSpatial >= 0.18
    ) {
        return {
            class: 'negative-or-gray-polarity',
            shortLabel: 'negative-gray',
            label: 'negative/gray polarity evidence',
            polarity: 'negative-or-gray',
            recommendedNextStep: 'investigate-polarity-aware-detection'
        };
    }

    if (
        confidentFrames > 0 &&
        likelyAbsentFrames >= halfFrameCount
    ) {
        return {
            class: 'intermittent-low-visible',
            shortLabel: 'intermittent',
            label: 'intermittent or low-visible watermark',
            polarity: negativeSpatialFrames > 0 ? 'mixed' : 'positive',
            recommendedNextStep: 'verify-frame-level-gate'
        };
    }

    if (likelyAbsentFrames >= strongFrameCount && meanAbsSpatial < 0.12) {
        return {
            class: 'likely-absent-or-off-anchor',
            shortLabel: 'likely-absent',
            label: 'likely absent or off-anchor',
            polarity: 'none',
            recommendedNextStep: 'verify-anchor-before-processing'
        };
    }

    return {
        class: 'ambiguous',
        shortLabel: 'ambiguous',
        label: 'ambiguous evidence',
        polarity: negativeSpatialFrames > positiveSpatialFrames ? 'mixed-negative' : 'mixed',
        recommendedNextStep: 'inspect-crop-sheet'
    };
}

function lumaAt(data, idx) {
    return 0.2126 * data[idx] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2];
}

function computeBackgroundMeanFromImageData(imageData, position, alphaMap, padding = 18) {
    const padX = Math.max(0, position.x - padding);
    const padY = Math.max(0, position.y - padding);
    const padRight = Math.min(imageData.width, position.x + position.width + padding);
    const padBottom = Math.min(imageData.height, position.y + position.height + padding);

    let sum = 0;
    let weightSum = 0;
    const lowAlphaThreshold = 0.015;

    for (let y = padY; y < padBottom; y++) {
        for (let x = padX; x < padRight; x++) {
            const inRoi =
                x >= position.x &&
                x < position.x + position.width &&
                y >= position.y &&
                y < position.y + position.height;

            let weight = inRoi ? 0 : 1;
            if (inRoi) {
                const rx = x - position.x;
                const ry = y - position.y;
                const alpha = alphaMap[ry * position.width + rx] || 0;
                if (alpha <= lowAlphaThreshold) weight = 0.35;
            }
            if (weight <= 0) continue;

            const idx = (y * imageData.width + x) * 4;
            sum += lumaAt(imageData.data, idx) * weight;
            weightSum += weight;
        }
    }

    return weightSum > 0 ? sum / weightSum : null;
}

function scoreGainAgainstBackground(imageData, position, alphaMap, gain, backgroundMean) {
    if (!Number.isFinite(backgroundMean)) return null;

    let sum = 0;
    let weightSum = 0;
    for (let y = 0; y < position.height; y++) {
        for (let x = 0; x < position.width; x++) {
            const rawAlpha = alphaMap[y * position.width + x] || 0;
            if (rawAlpha <= 0.025) continue;

            const alpha = Math.min(rawAlpha * gain, 0.99);
            const oneMinusAlpha = 1 - alpha;
            const idx = ((position.y + y) * imageData.width + position.x + x) * 4;
            const weight = Math.min(1, Math.max(0, rawAlpha * 8));

            const r = (imageData.data[idx] - alpha * LOGO_VALUE) / oneMinusAlpha;
            const g = (imageData.data[idx + 1] - alpha * LOGO_VALUE) / oneMinusAlpha;
            const b = (imageData.data[idx + 2] - alpha * LOGO_VALUE) / oneMinusAlpha;
            sum += (0.2126 * r + 0.7152 * g + 0.0722 * b) * weight;
            weightSum += weight;
        }
    }

    if (weightSum <= 0) return null;
    return sum / weightSum - backgroundMean;
}

function estimateFrameAlphaGain(imageData, position, alphaMap, seedGain = DEFAULT_ALPHA_SEED_GAIN) {
    const backgroundMean = computeBackgroundMeanFromImageData(imageData, position, alphaMap);
    if (!Number.isFinite(backgroundMean)) return null;

    let lo = Math.max(0.35, seedGain - 0.45);
    let hi = Math.min(1.35, seedGain + 0.45);
    let bestGain = seedGain;
    let bestAbsDelta = Number.POSITIVE_INFINITY;

    for (let i = 0; i < ALPHA_REFINEMENT_ROUNDS; i++) {
        const gain = (lo + hi) / 2;
        const delta = scoreGainAgainstBackground(imageData, position, alphaMap, gain, backgroundMean);
        if (!Number.isFinite(delta)) return null;

        const absDelta = Math.abs(delta);
        if (absDelta < bestAbsDelta) {
            bestAbsDelta = absDelta;
            bestGain = gain;
        }

        if (delta > 0) {
            lo = gain;
        } else {
            hi = gain;
        }
    }

    return {
        gain: bestGain,
        residualDelta: bestAbsDelta
    };
}

function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
}

function estimateAlphaSeedFromFrames(frames, position, alphaMap, frameWinners, candidateId) {
    const winnerByTimestamp = new Map(frameWinners.map((winner) => [winner.timestamp, winner]));
    const estimates = [];

    for (const frame of frames) {
        const winner = winnerByTimestamp.get(frame.timestamp);
        if (winner && winner.candidateId !== candidateId && winner.confidence > 0.08) continue;

        const estimate = estimateFrameAlphaGain(frame.imageData, position, alphaMap);
        if (!estimate) continue;
        if (estimate.gain < 0.35 || estimate.gain > 1.35) continue;
        estimates.push(estimate.gain);
    }

    const seedGain = median(estimates) ?? DEFAULT_ALPHA_SEED_GAIN;
    return {
        seedGain,
        estimates,
        estimateCount: estimates.length
    };
}

function summarizeCandidate(scores) {
    if (!scores.length) {
        return {
            frames: 0,
            meanSpatial: 0,
            meanGradient: 0,
            meanConfidence: 0,
            maxConfidence: 0,
            votes: 0
        };
    }

    const sum = scores.reduce((acc, score) => {
        acc.spatial += score.spatial;
        acc.gradient += score.gradient;
        acc.confidence += score.confidence;
        acc.maxConfidence = Math.max(acc.maxConfidence, score.confidence);
        return acc;
    }, { spatial: 0, gradient: 0, confidence: 0, maxConfidence: 0 });

    return {
        frames: scores.length,
        meanSpatial: sum.spatial / scores.length,
        meanGradient: sum.gradient / scores.length,
        meanConfidence: sum.confidence / scores.length,
        maxConfidence: sum.maxConfidence,
        votes: 0
    };
}

export function selectIndependentVideoWatermarkTracks(summaries, {
    minConfidence = DEFAULT_MIN_CONFIDENCE
} = {}) {
    if (!Array.isArray(summaries) || summaries.length === 0) return [];

    const qualified = summaries.filter((summary, index) => {
        if (index === 0) return true;
        // When every sample agrees on the primary, a zero-vote background track
        // must not displace it through per-frame confidence normalization.
        const primary = summaries[0];
        const unanimousPrimary = primary.frames > 0 && primary.votes === primary.frames;
        return summary.meanConfidence >= minConfidence && (!unanimousPrimary || summary.votes > 0);
    });
    const selected = [];

    for (const summary of qualified) {
        const candidate = summary.candidate;
        const candidateSize = Number(candidate?.size);
        const candidateArea = candidateSize * candidateSize;
        const overlapsSelected = selected.some((selectedSummary) => {
            const selectedCandidate = selectedSummary.candidate;
            const selectedSize = Number(selectedCandidate?.size);
            const overlapWidth = Math.max(0, Math.min(
                Number(candidate?.x) + candidateSize,
                Number(selectedCandidate?.x) + selectedSize
            ) - Math.max(Number(candidate?.x), Number(selectedCandidate?.x)));
            const overlapHeight = Math.max(0, Math.min(
                Number(candidate?.y) + candidateSize,
                Number(selectedCandidate?.y) + selectedSize
            ) - Math.max(Number(candidate?.y), Number(selectedCandidate?.y)));
            const smallerArea = Math.min(candidateArea, selectedSize * selectedSize);
            return smallerArea > 0 && (overlapWidth * overlapHeight) / smallerArea >= 0.5;
        });
        if (!overlapsSelected) selected.push(summary);
    }

    return selected;
}

export function selectVideoWatermarkDetectionForFrame(imageSource, detection) {
    const detections = Array.isArray(detection?.detections) && detection.detections.length > 0
        ? detection.detections
        : [detection];
    let selected = null;

    for (const candidateDetection of detections) {
        if (!candidateDetection?.position || !candidateDetection?.alphaMap) continue;
        const position = candidateDetection.position;
        const hasRoiReader = typeof imageSource?.getImageData === 'function';
        const imageData = hasRoiReader
            ? imageSource.getImageData(position.x, position.y, position.width, position.height)
            : imageSource;
        const frameScore = scoreVideoWatermarkFrame(
            imageData,
            hasRoiReader
                ? { x: 0, y: 0, width: position.width, height: position.height }
                : position,
            candidateDetection.alphaMap
        );
        const meanConfidence = Math.max(0.01, Number(candidateDetection.meanConfidence) || 0.01);
        const normalizedConfidence = frameScore.confidence / meanConfidence;
        if (
            !selected ||
            normalizedConfidence > selected.normalizedConfidence ||
            (normalizedConfidence === selected.normalizedConfidence && frameScore.confidence > selected.frameScore.confidence)
        ) {
            selected = {
                detection: candidateDetection,
                frameScore,
                normalizedConfidence
            };
        }
    }

    return selected;
}

export function detectDiamondVideoWatermarkFromFrames({
    frames,
    width,
    height,
    candidates = resolveVideoWatermarkCandidates(width, height),
    minConfidence = DEFAULT_MIN_CONFIDENCE,
    alphaMapOptions = {}
}) {
    if (!Array.isArray(frames) || frames.length === 0) {
        throw new Error('没有可用于检测的视频帧');
    }
    if (!candidates.length) {
        throw new Error(`暂不支持 ${width}x${height} 的视频水印候选`);
    }

    const perCandidate = new Map(candidates.map((candidate) => [candidate.id, {
        candidate,
        scores: []
    }]));

    const frameWinners = [];
    for (const frame of frames) {
        const scored = candidates
            .map((candidate) => scoreCandidateOnFrame(frame.imageData, candidate, alphaMapOptions))
            .sort((a, b) => b.confidence - a.confidence);

        const winner = scored[0] || null;
        if (winner) {
            frameWinners.push({
                timestamp: frame.timestamp,
                candidateId: winner.candidate.id,
                confidence: winner.confidence
            });
        }

        for (const score of scored) {
            perCandidate.get(score.candidate.id).scores.push(score);
        }
    }

    for (const winner of frameWinners) {
        const entry = perCandidate.get(winner.candidateId);
        if (entry) entry.votes = (entry.votes || 0) + 1;
    }

    const summaries = [...perCandidate.values()]
        .map((entry) => ({
            candidate: entry.candidate,
            ...summarizeCandidate(entry.scores),
            votes: entry.votes || 0
        }))
        .sort((a, b) => {
            if (b.votes !== a.votes) return b.votes - a.votes;
            return b.meanConfidence - a.meanConfidence;
        });

    const best = summaries[0];
    const trackSummaries = selectIndependentVideoWatermarkTracks(summaries, { minConfidence });
    const voteRatio = frames.length > 0 ? best.votes / frames.length : 0;
    const combinedVoteRatio = frames.length > 0
        ? Math.min(1, trackSummaries.reduce((sum, summary) => sum + summary.votes, 0) / frames.length)
        : 0;
    const isConfident =
        best.meanConfidence >= minConfidence &&
        (voteRatio >= 0.6 || (trackSummaries.length > 1 && combinedVoteRatio >= 0.6));
    const position = {
        x: best.candidate.x,
        y: best.candidate.y,
        width: best.candidate.size,
        height: best.candidate.size
    };
    const alphaSelection = selectVideoAlphaShapeForDetection({
        frames,
        position,
        candidate: best.candidate,
        best,
        voteRatio,
        frameWinners,
        alphaMapOptions
    });
    const { alphaMap, alphaSeed } = alphaSelection;
    const detections = [{
        watermarkKind: 'diamond',
        position,
        alphaMap,
        alphaSeed,
        candidate: best.candidate,
        meanConfidence: best.meanConfidence
    }];

    for (const summary of trackSummaries.slice(1)) {
        const candidatePosition = {
            x: summary.candidate.x,
            y: summary.candidate.y,
            width: summary.candidate.size,
            height: summary.candidate.size
        };
        const candidateAlphaSelection = selectVideoAlphaShapeForDetection({
            frames,
            position: candidatePosition,
            candidate: summary.candidate,
            best: summary,
            voteRatio: frames.length > 0 ? summary.votes / frames.length : 0,
            frameWinners,
            alphaMapOptions
        });
        detections.push({
            watermarkKind: 'diamond',
            position: candidatePosition,
            alphaMap: candidateAlphaSelection.alphaMap,
            alphaSeed: candidateAlphaSelection.alphaSeed,
            candidate: summary.candidate,
            meanConfidence: summary.meanConfidence
        });
    }

    return {
        watermarkKind: 'diamond',
        position,
        alphaMap,
        alphaSeed,
        candidate: best.candidate,
        detections,
        isConfident,
        summary: {
            frameCount: frames.length,
            minConfidence,
            alphaSeed,
            alphaShape: alphaSelection.alphaShape,
            best: {
                candidateId: best.candidate.id,
                label: best.candidate.label,
                meanSpatial: best.meanSpatial,
                meanGradient: best.meanGradient,
                meanConfidence: best.meanConfidence,
                maxConfidence: best.maxConfidence,
                votes: best.votes
            },
            candidates: summaries.map((summary) => ({
                candidateId: summary.candidate.id,
                label: summary.candidate.label,
                x: summary.candidate.x,
                y: summary.candidate.y,
                size: summary.candidate.size,
                meanSpatial: summary.meanSpatial,
                meanGradient: summary.meanGradient,
                meanConfidence: summary.meanConfidence,
                maxConfidence: summary.maxConfidence,
                votes: summary.votes
            })),
            frameWinners
        }
    };
}

async function maybeYieldToMainThread(yieldToMainThread) {
    if (typeof yieldToMainThread === 'function') {
        await yieldToMainThread();
    }
}

export async function detectDiamondVideoWatermarkFromFramesAsync({
    frames,
    width,
    height,
    candidates = resolveVideoWatermarkCandidates(width, height),
    minConfidence = DEFAULT_MIN_CONFIDENCE,
    alphaMapOptions = {},
    yieldToMainThread = null
}) {
    if (!Array.isArray(frames) || frames.length === 0) {
        throw new Error('No video frames are available for detection');
    }
    if (!candidates.length) {
        throw new Error('Unsupported video watermark candidates for ' + width + 'x' + height);
    }

    const perCandidate = new Map(candidates.map((candidate) => [candidate.id, {
        candidate,
        scores: []
    }]));

    const frameWinners = [];
    for (const frame of frames) {
        const scored = candidates
            .map((candidate) => scoreCandidateOnFrame(frame.imageData, candidate, alphaMapOptions))
            .sort((a, b) => b.confidence - a.confidence);

        const winner = scored[0] || null;
        if (winner) {
            frameWinners.push({
                timestamp: frame.timestamp,
                candidateId: winner.candidate.id,
                confidence: winner.confidence
            });
        }

        for (const score of scored) {
            perCandidate.get(score.candidate.id).scores.push(score);
        }
        await maybeYieldToMainThread(yieldToMainThread);
    }

    for (const winner of frameWinners) {
        const entry = perCandidate.get(winner.candidateId);
        if (entry) entry.votes = (entry.votes || 0) + 1;
    }

    const summaries = [...perCandidate.values()]
        .map((entry) => ({
            candidate: entry.candidate,
            ...summarizeCandidate(entry.scores),
            votes: entry.votes || 0
        }))
        .sort((a, b) => {
            if (b.votes !== a.votes) return b.votes - a.votes;
            return b.meanConfidence - a.meanConfidence;
        });

    const best = summaries[0];
    const trackSummaries = selectIndependentVideoWatermarkTracks(summaries, { minConfidence });
    const voteRatio = frames.length > 0 ? best.votes / frames.length : 0;
    const combinedVoteRatio = frames.length > 0
        ? Math.min(1, trackSummaries.reduce((sum, summary) => sum + summary.votes, 0) / frames.length)
        : 0;
    const isConfident =
        best.meanConfidence >= minConfidence &&
        (voteRatio >= 0.6 || (trackSummaries.length > 1 && combinedVoteRatio >= 0.6));
    const position = {
        x: best.candidate.x,
        y: best.candidate.y,
        width: best.candidate.size,
        height: best.candidate.size
    };
    const alphaSelection = selectVideoAlphaShapeForDetection({
        frames,
        position,
        candidate: best.candidate,
        best,
        voteRatio,
        frameWinners,
        alphaMapOptions
    });
    const { alphaMap, alphaSeed } = alphaSelection;
    const detections = [{
        watermarkKind: 'diamond',
        position,
        alphaMap,
        alphaSeed,
        candidate: best.candidate,
        meanConfidence: best.meanConfidence
    }];

    for (const summary of trackSummaries.slice(1)) {
        const candidatePosition = {
            x: summary.candidate.x,
            y: summary.candidate.y,
            width: summary.candidate.size,
            height: summary.candidate.size
        };
        const candidateAlphaSelection = selectVideoAlphaShapeForDetection({
            frames,
            position: candidatePosition,
            candidate: summary.candidate,
            best: summary,
            voteRatio: frames.length > 0 ? summary.votes / frames.length : 0,
            frameWinners,
            alphaMapOptions
        });
        detections.push({
            watermarkKind: 'diamond',
            position: candidatePosition,
            alphaMap: candidateAlphaSelection.alphaMap,
            alphaSeed: candidateAlphaSelection.alphaSeed,
            candidate: summary.candidate,
            meanConfidence: summary.meanConfidence
        });
    }

    return {
        watermarkKind: 'diamond',
        position,
        alphaMap,
        alphaSeed,
        candidate: best.candidate,
        detections,
        isConfident,
        summary: {
            frameCount: frames.length,
            minConfidence,
            alphaSeed,
            alphaShape: alphaSelection.alphaShape,
            best: {
                candidateId: best.candidate.id,
                label: best.candidate.label,
                meanSpatial: best.meanSpatial,
                meanGradient: best.meanGradient,
                meanConfidence: best.meanConfidence,
                maxConfidence: best.maxConfidence,
                votes: best.votes
            },
            candidates: summaries.map((summary) => ({
                candidateId: summary.candidate.id,
                label: summary.candidate.label,
                x: summary.candidate.x,
                y: summary.candidate.y,
                size: summary.candidate.size,
                meanSpatial: summary.meanSpatial,
                meanGradient: summary.meanGradient,
                meanConfidence: summary.meanConfidence,
                maxConfidence: summary.maxConfidence,
                votes: summary.votes
            })),
            frameWinners
        }
    };
}

export function selectVideoWatermarkDetection({
    diamondDetection = null,
    veoTextDetection = null,
    diamondWeakConfidenceCeiling = 0.28
} = {}) {
    const diamondBestConfidence = diamondDetection?.summary?.best?.meanConfidence ?? 0;
    const veoBestNcc = veoTextDetection?.summary?.best?.meanNcc ?? 0;
    const veoStrong = veoTextDetection?.isConfident === true;
    const diamondStrong = diamondDetection?.isConfident === true;

    if (veoStrong && (!diamondStrong || diamondBestConfidence < diamondWeakConfidenceCeiling || veoBestNcc > diamondBestConfidence)) {
        return {
            ...veoTextDetection,
            summary: {
                ...veoTextDetection.summary,
                alternatives: {
                    diamond: diamondDetection?.summary?.best ?? null
                }
            }
        };
    }

    if (diamondDetection) {
        return {
            ...diamondDetection,
            summary: {
                ...diamondDetection.summary,
                alternatives: {
                    veoText: veoTextDetection?.summary?.best ?? null,
                    veoTextCandidates: veoTextDetection?.summary?.candidates ?? []
                }
            }
        };
    }

    return veoTextDetection;
}

export function detectVideoWatermarkFromFrames({
    frames,
    width,
    height,
    candidates = resolveVideoWatermarkCandidates(width, height),
    minConfidence = DEFAULT_MIN_CONFIDENCE,
    alphaMapOptions = {},
    veoTextOptions = {}
}) {
    let diamondDetection = null;
    if (Array.isArray(candidates) && candidates.length > 0) {
        diamondDetection = detectDiamondVideoWatermarkFromFrames({
            frames,
            width,
            height,
            candidates,
            minConfidence,
            alphaMapOptions
        });
    }

    const veoTextDetection = detectVeoTextWatermarkFromFrames({
        frames,
        width,
        height,
        ...veoTextOptions
    });

    const selected = selectVideoWatermarkDetection({
        diamondDetection,
        veoTextDetection
    });

    if (!selected) {
        if (!Array.isArray(candidates) || candidates.length === 0) {
            throw new Error('Unsupported video watermark candidates for ' + width + 'x' + height);
        }
        throw new Error('No video watermark detection result is available');
    }

    return selected;
}

export async function detectVideoWatermarkFromFramesAsync({
    frames,
    width,
    height,
    candidates = resolveVideoWatermarkCandidates(width, height),
    minConfidence = DEFAULT_MIN_CONFIDENCE,
    alphaMapOptions = {},
    veoTextOptions = {},
    yieldToMainThread = null
}) {
    let diamondDetection = null;
    if (Array.isArray(candidates) && candidates.length > 0) {
        diamondDetection = await detectDiamondVideoWatermarkFromFramesAsync({
            frames,
            width,
            height,
            candidates,
            minConfidence,
            alphaMapOptions,
            yieldToMainThread
        });
    }

    const veoTextDetection = await detectVeoTextWatermarkFromFramesAsync({
        frames,
        width,
        height,
        ...veoTextOptions,
        yieldToMainThread
    });

    const selected = selectVideoWatermarkDetection({
        diamondDetection,
        veoTextDetection
    });

    if (!selected) {
        if (!Array.isArray(candidates) || candidates.length === 0) {
            throw new Error('Unsupported video watermark candidates for ' + width + 'x' + height);
        }
        throw new Error('No video watermark detection result is available');
    }

    return selected;
}

export { getVideoAlphaMap };

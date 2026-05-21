%% Project 2: Modulation Classification with Deep Learning
%默认 trainNow = false，直接用的是老师给的 trainedModulationClassificationNetwork.mat。
%如果想重新训练并显示 training-progress 曲线，把 trainNow 改成 true。

clear; clc; close all;

%% 0. 基本设置
scriptFolder = fileparts(mfilename('fullpath'));
if ~isempty(scriptFolder)
    cd(scriptFolder);
    addpath(scriptFolder);
end

rng(1235);

modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
    "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
    "B-FM", "DSB-AM", "SSB-AM"]);

% 报告重点观察的三种调制方式
focusModTypes = categorical(["QPSK", "8PSK", "64QAM"]);

sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
fs = 200e3;             % Sample rate
SNR = 30;               % AWGN SNR, dB
transDelay = 50;        % 去除滤波和信道开始处的暂态

% 数据划分要求：训练集 70%，验证集 15%，测试集 15%
percentTrainingSamples = 70;
percentValidationSamples = 15;
percentTestSamples = 15;

numFramesPerModType = 200;

% false：直接加载训练好的网络
trainNow = false;
maxEpochs = 12;
miniBatchSize = 256;

outputFolder = fullfile(pwd, "Project2_OutputFigures");
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

dataDirectory = fullfile(pwd, "Project2_GeneratedFrames");

%% 加载或训练 CNN 网络
fprintf("\n========== Part 1: Load / Train CNN ==========" + newline);

if trainNow
    fprintf("开始生成训练、验证、测试数据，并重新训练 CNN。\n");
    [rxFrames, rxLabels] = generateDatasetInMemory(modulationTypes, numFramesPerModType, ...
        sps, spf, fs, SNR, transDelay, dataDirectory);

    [trainIdx, validIdx, testIdx] = splitByLabel(rxLabels, ...
        percentTrainingSamples, percentValidationSamples, percentTestSamples);

    rxTrainFrames = rxFrames(:,:,:,trainIdx);
    rxValidFrames = rxFrames(:,:,:,validIdx);
    rxTestFrames  = rxFrames(:,:,:,testIdx);

    rxTrainLabels = rxLabels(trainIdx);
    rxValidLabels = rxLabels(validIdx);
    rxTestLabels  = rxLabels(testIdx);

    modClassNet = helperModClassCNN(modulationTypes, sps, spf);
    options = helperModClassTrainingOptions(maxEpochs, miniBatchSize, ...
        numel(rxTrainLabels), rxValidFrames, rxValidLabels);

    trainedNet = trainNetwork(rxTrainFrames, rxTrainLabels, modClassNet, options);
    save("trainedNet_Project2_70_15_15.mat", "trainedNet");
else
    fprintf("使用老师提供的 trainedModulationClassificationNetwork.mat。\n");
    tmp = load("trainedModulationClassificationNetwork.mat");
    if isfield(tmp, "trainedNet")
        trainedNet = tmp.trainedNet;
    else
        names = fieldnames(tmp);
        trainedNet = tmp.(names{1});
    end
end

%% 基本任务1：QPSK / 8PSK / 64QAM 分类概率图
fprintf("\n========== Part 2: Classification Scores for QPSK / 8PSK / 64QAM ==========" + newline);

scoreSummary = table;
for k = 1:numel(focusModTypes)
    modType = focusModTypes(k);
    [predictedLabels, scores] = classifyOneModulation(trainedNet, modType, modulationTypes, ...
        sps, spf, fs, SNR);

    [bestScore, bestIdx] = max(scores, [], 2);
    classNames = categories(modulationTypes);
    bestName = categorical(string(classNames(bestIdx)), string(classNames));

    T = table((1:numel(predictedLabels))', predictedLabels(:), bestName(:), bestScore(:), ...
        'VariableNames', {'Frame', 'ClassifyResult', 'MaxScoreClass', 'MaxScore'});
    writetable(T, fullfile(outputFolder, "classification_" + string(modType) + ".csv"));
    disp("调制方式 " + string(modType) + " 的分类结果：");
    disp(T);

    T2 = addvars(T, repmat(string(modType), height(T), 1), ...
        'Before', 'Frame', 'NewVariableNames', 'TrueModulation');
    if isempty(scoreSummary)
        scoreSummary = T2;
    else
        scoreSummary = [scoreSummary; T2]; %#ok<AGROW>
    end

    fig = figure('Name', "Classification Scores - " + string(modType));
    plotScoreBars(scores, modulationTypes, "Classification Scores - " + string(modType));
    saveFigureSafe(fig, fullfile(outputFolder, "score_" + string(modType) + ".png"));
end
writetable(scoreSummary, fullfile(outputFolder, "classification_score_summary.csv"));

%% 基本任务3：使用 filter 和 FFT 画 QPSK / 8PSK / 64QAM 的时域图与频谱图
fprintf("\n========== Part 3: filter Output and FFT Spectrum ==========" + newline);
plotFilteredSignalExamples(focusModTypes, sps, fs, outputFolder);

%% 扩展任务1：生成比例为 70 / 15 / 15 的数据集并测试混淆矩阵
% 若上面已经 trainNow = true，这里会直接复用训练时生成的数据；否则这里会生成一个比较小的测试数据集，用给的的 trainedNet 进行测试，并画混淆矩阵。
fprintf("\n========== Part 4: 70/15/15 Split and Confusion Matrix ==========" + newline);

if ~trainNow
    [rxFrames, rxLabels] = generateDatasetInMemory(modulationTypes, numFramesPerModType, ...
        sps, spf, fs, SNR, transDelay, dataDirectory);
    [trainIdx, validIdx, testIdx] = splitByLabel(rxLabels, ...
        percentTrainingSamples, percentValidationSamples, percentTestSamples);
    rxTrainLabels = rxLabels(trainIdx);
    rxValidLabels = rxLabels(validIdx);
    rxTestFrames = rxFrames(:,:,:,testIdx);
    rxTestLabels = rxLabels(testIdx);
end

rxTestPred = classify(trainedNet, rxTestFrames);
testAccuracy = mean(rxTestPred == rxTestLabels);
fprintf("Test accuracy = %.2f%%\n", 100 * testAccuracy);

fig = figure('Name', 'Confusion Matrix for Test Data');
cm = confusionchart(rxTestLabels, rxTestPred);
cm.Title = sprintf('Confusion Matrix for Test Data, Accuracy = %.2f%%', 100 * testAccuracy);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveFigureSafe(fig, fullfile(outputFolder, "confusion_matrix_test.png"));

%% =========================== 使用的局部函数 ===========================

function [rxFrames, rxLabels] = generateDatasetInMemory(modulationTypes, numFramesPerModType, ...
    sps, spf, fs, SNR, transDelay, dataDirectory)

    if ~exist(dataDirectory, 'dir')
        mkdir(dataDirectory);
    end

    numModTypes = numel(modulationTypes);
    totalFrames = numModTypes * numFramesPerModType;
    rxFrames = zeros(1, spf, 2, totalFrames, 'single');
    rxLabels = categorical(repmat(string(modulationTypes(1)), totalFrames, 1));
    rxLabels = setcats(rxLabels, categories(modulationTypes));

    frameCnt = 1;
    for i = 1:numModTypes
        modType = modulationTypes(i);
        fprintf("Generating %s frames...\n", string(modType));

        dataSrc = helperModClassGetSource(string(modType), sps, 2*spf, fs);
        modulator = helperModClassGetModulator(string(modType), sps, fs);

        if contains(char(modType), {'B-FM','DSB-AM','SSB-AM'})
            centerFrequency = 100e6;
        else
            centerFrequency = 902e6;
        end

        for p = 1:numFramesPerModType
            x = dataSrc();
            tx = modulator(x);
            rx = applyChannelImpairments(tx, fs, SNR, centerFrequency);
            oneFrame = helperModClassFrameGenerator(rx, spf, spf, transDelay, sps);
            oneFrame = oneFrame(:,1);

            rxFrames(:,:,:,frameCnt) = single(complexFrameToCNNInput(oneFrame));
            rxLabels(frameCnt) = modType;

            fileName = fullfile(dataDirectory, sprintf("frame%s%04d.mat", string(modType), p));
            frame = oneFrame;
            label = modType;
            save(fileName, "frame", "label");

            frameCnt = frameCnt + 1;
        end
    end
end

function [trainIdx, validIdx, testIdx] = splitByLabel(labels, percentTrain, percentValid, percentTest)
    if abs(percentTrain + percentValid + percentTest - 100) > 1e-9
        error("训练集、验证集、测试集比例之和必须为 100。");
    end

    cats = categories(labels);
    trainIdx = [];
    validIdx = [];
    testIdx = [];

    for i = 1:numel(cats)
        idx = find(string(labels) == string(cats{i}));
        idx = idx(randperm(numel(idx)));

        n = numel(idx);
        nTrain = floor(n * percentTrain / 100);
        nValid = floor(n * percentValid / 100);

        trainIdx = [trainIdx; idx(1:nTrain)]; %#ok<AGROW>
        validIdx = [validIdx; idx(nTrain+1:nTrain+nValid)]; %#ok<AGROW>
        testIdx  = [testIdx;  idx(nTrain+nValid+1:end)]; %#ok<AGROW>
    end

    trainIdx = trainIdx(randperm(numel(trainIdx)));
    validIdx = validIdx(randperm(numel(validIdx)));
    testIdx = testIdx(randperm(numel(testIdx)));
end

function [predictedLabels, scores] = classifyOneModulation(trainedNet, modType, modulationTypes, ...
    sps, spf, fs, SNR)

    numDemoSamples = 8 * spf;
    dataSrc = helperModClassGetSource(string(modType), sps, numDemoSamples, fs);
    modulator = helperModClassGetModulator(string(modType), sps, fs);

    if contains(char(modType), {'B-FM','DSB-AM','SSB-AM'})
        centerFrequency = 100e6;
    else
        centerFrequency = 902e6;
    end

    x = dataSrc();
    tx = modulator(x);
    rx = applyChannelImpairments(tx, fs, SNR, centerFrequency);

    unknownFrames = helperModClassGetNNFrames(rx);
    [predictedLabels, scores] = classify(trainedNet, unknownFrames);

    % 如果网络输出类别和 modulationTypes 顺序不一致，只做基本提醒。
    if size(scores,2) ~= numel(modulationTypes)
        warning("score 列数和 modulationTypes 数量不一致，请检查网络输出类别。")
    end
end

function rx = applyChannelImpairments(tx, fs, SNR, centerFrequency)
    % Rician multipath fading + clock offset 导致的频偏和采样时间漂移 + AWGN
    tx = tx(:);

    multipathChannel = comm.RicianChannel( ...
        'SampleRate', fs, ...
        'PathDelays', [0 1.8 3.4] / fs, ...
        'AveragePathGains', [0 -2 -10], ...
        'KFactor', 4, ...
        'MaximumDopplerShift', 4);
    reset(multipathChannel);
    y = multipathChannel(tx);

    maxClockOffset = 5;                 % ppm
    clockOffset = (rand()*2*maxClockOffset) - maxClockOffset;
    C = 1 + clockOffset / 1e6;

    frequencyShifter = comm.PhaseFrequencyOffset('SampleRate', fs);
    frequencyShifter.FrequencyOffset = -(C - 1) * centerFrequency;
    y = frequencyShifter(y);

    t = (0:length(y)-1)' / fs;
    newFs = fs * C;
    tp = (0:length(y)-1)' / newFs;
    y = interp1(t, y, tp, 'linear', 'extrap');

    rx = awgn(y, SNR, 0);
end

function frameReal = complexFrameToCNNInput(frameComplex)
    I = permute(real(frameComplex), [3 1 4 2]);
    Q = permute(imag(frameComplex), [3 1 4 2]);
    frameReal = cat(3, I, Q);
end

function plotScoreBars(scores, modulationTypes, figTitle)
    classNames = categories(modulationTypes);
    bar(scores, 'stacked');
    grid on;
    xlabel('Frame Number');
    ylabel('Score');
    title(figTitle);
    legend(classNames, 'Location', 'eastoutside');
    ylim([0 1]);
end

function plotFilteredSignalExamples(focusModTypes, sps, fs, outputFolder)
    filterCoeffs = rcosdesign(0.35, 4, sps);

    for k = 1:numel(focusModTypes)
        modType = string(focusModTypes(k));
        numSymbols = 1024;

        switch modType
            case "QPSK"
                d = randi([0 3], numSymbols, 1);
                syms = pskmod(d, 4, pi/4);
            case "8PSK"
                d = randi([0 7], numSymbols, 1);
                syms = pskmod(d, 8);
            case "64QAM"
                d = randi([0 63], numSymbols, 1);
                syms = qammod(d, 64, 'UnitAveragePower', true);
            otherwise
                error("当前函数只处理 QPSK / 8PSK / 64QAM。");
        end

        tx = filter(filterCoeffs, 1, upsample(syms, sps));
        t = 1000 * (0:length(tx)-1)' / fs;

        nfft = 4096;
        f = (-nfft/2:nfft/2-1) * fs / nfft / 1000;
        spec = fftshift(abs(fft(tx, nfft)));
        spec = spec / max(spec);

        fig = figure('Name', "filter and FFT - " + modType);
        subplot(2,1,1);
        plot(t, real(tx)); hold on;
        plot(t, imag(tx)); hold off;
        grid on;
        xlabel('Time (ms)');
        ylabel('Amplitude');
        title(modType + " after pulse-shaping filter");
        legend('I component', 'Q component');

        subplot(2,1,2);
        plot(f, spec);
        grid on;
        xlabel('Frequency (kHz)');
        ylabel('Normalized Magnitude');
        title(modType + " FFT Spectrum");

        saveFigureSafe(fig, fullfile(outputFolder, "filter_fft_" + modType + ".png"));
    end
end

function saveFigureSafe(fig, fileName)
    try
        exportgraphics(fig, fileName, 'Resolution', 180);
    catch
        saveas(fig, fileName);
    end
end

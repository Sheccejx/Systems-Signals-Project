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
trainNow = true;
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

%% 扩展任务2：SNR对分类准确率的定量影响分析
fprintf("\n========== Part 5: SNR-Accuracy Relationship Analysis ==========" + newline);

% 1. 定义要测试的SNR范围
SNR_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30]; % dB
% SNR_list = -10:5:30; % 或者使用这个范围，从-10dB到30dB，步长5dB

% 2. 为每个SNR创建测试集并评估准确率
numTestFramesPerSNR = 200; % 每个SNR下的总测试帧数（所有调制类型）
numModTypes = numel(modulationTypes);
testFramesPerModPerSNR = floor(numTestFramesPerSNR / numModTypes);

% 预存储结果
accuracies = zeros(size(SNR_list));
confusionMatrices = cell(size(SNR_list));
allTestLabels = {};
allTestPreds = {};

% 创建输出目录
snrOutputFolder = fullfile(outputFolder, 'SNR_Analysis');
if ~exist(snrOutputFolder, 'dir')
    mkdir(snrOutputFolder);
end

% 进度显示
fprintf("开始SNR影响分析，共 %d 个SNR点...\n", numel(SNR_list));

for snr_idx = 1:numel(SNR_list)
    current_SNR = SNR_list(snr_idx);
    fprintf("正在处理 SNR = %d dB (%d/%d)...\n", current_SNR, snr_idx, numel(SNR_list));
    
    % 生成当前SNR下的测试数据
    [testFrames_snr, testLabels_snr] = generateTestDatasetForSNR(...
        modulationTypes, testFramesPerModPerSNR, sps, spf, fs, current_SNR, transDelay);
    
    % 使用训练好的网络进行预测
    testPred_snr = classify(trainedNet, testFrames_snr);
    
    % 计算准确率
    accuracy_snr = mean(testPred_snr == testLabels_snr);
    accuracies(snr_idx) = accuracy_snr;
    
    % 存储结果用于后续分析
    allTestLabels{snr_idx} = testLabels_snr;
    allTestPreds{snr_idx} = testPred_snr;
    
    % 创建当前SNR的混淆矩阵
    fig_cm = figure('Visible', 'off');
    cm = confusionchart(testLabels_snr, testPred_snr);
    cm.Title = sprintf('Confusion Matrix at SNR = %d dB, Accuracy = %.2f%%', ...
        current_SNR, 100 * accuracy_snr);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    
    % 保存混淆矩阵图像
    saveFigureSafe(fig_cm, fullfile(snrOutputFolder, ...
        sprintf('confusion_matrix_SNR_%d.png', current_SNR)));
    close(fig_cm);
    
    fprintf("  SNR = %d dB: 准确率 = %.2f%%\n", current_SNR, 100 * accuracy_snr);
end

% 3. 绘制SNR-准确率曲线
fig_snr = figure('Name', 'SNR vs. Classification Accuracy', 'Position', [100, 100, 800, 600]);
plot(SNR_list, 100 * accuracies, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Classification Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('SNR对调制分类准确率的影响', 'FontSize', 14, 'FontWeight', 'bold');
xlim([min(SNR_list)-1, max(SNR_list)+1]);
ylim([0, 105]);

% 在每个数据点上添加准确率文本标签
for i = 1:numel(SNR_list)
    text(SNR_list(i), 100 * accuracies(i) + 1, ...
        sprintf('%.1f%%', 100 * accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 添加参考线
hold on;
plot([min(SNR_list), max(SNR_list)], [50, 50], 'r--', 'LineWidth', 1, 'DisplayName', '50% Reference');
plot([min(SNR_list), max(SNR_list)], [90, 90], 'g--', 'LineWidth', 1, 'DisplayName', '90% Reference');
legend('SNR-Accuracy Curve', '50% Reference', '90% Reference', 'Location', 'southeast');
hold off;

% 保存图像
saveFigureSafe(fig_snr, fullfile(snrOutputFolder, 'SNR_vs_Accuracy.png'));

% 4. 绘制每种调制方式在不同SNR下的准确率
fprintf("\n分析每种调制方式在不同SNR下的表现...\n");

% 计算每种调制方式在不同SNR下的准确率
modAccuracies = zeros(numModTypes, numel(SNR_list));
modNames = string(modulationTypes);

for snr_idx = 1:numel(SNR_list)
    current_labels = allTestLabels{snr_idx};
    current_preds = allTestPreds{snr_idx};
    
    for mod_idx = 1:numModTypes
        mod = modulationTypes(mod_idx);
        idx = (current_labels == mod);
        if any(idx)
            modAcc = mean(current_preds(idx) == current_labels(idx));
            modAccuracies(mod_idx, snr_idx) = modAcc;
        end
    end
end

% 绘制每种调制方式的SNR-准确率曲线
fig_mods = figure('Name', 'SNR vs. Accuracy per Modulation Type', 'Position', [100, 100, 1000, 700]);
colors = lines(numModTypes); % 使用不同的颜色
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '*', '+'};

hold on;
for mod_idx = 1:numModTypes
    plot(SNR_list, 100 * modAccuracies(mod_idx, :), ...
        'Color', colors(mod_idx, :), ...
        'Marker', markers{mod_idx}, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6, ...
        'DisplayName', char(modulationTypes(mod_idx)));
end
hold off;

grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Classification Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('不同调制方式在不同SNR下的分类准确率', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 10);
xlim([min(SNR_list)-1, max(SNR_list)+1]);
ylim([0, 105]);

% 保存图像
saveFigureSafe(fig_mods, fullfile(snrOutputFolder, 'SNR_vs_Accuracy_per_Modulation.png'));

% 5. 创建结果汇总表并保存
resultsTable = table(SNR_list', 100 * accuracies', ...
    'VariableNames', {'SNR_dB', 'Overall_Accuracy_Percent'});

% 为每种调制方式添加列
for mod_idx = 1:numModTypes
    modName = char(modulationTypes(mod_idx));
    modNameClean = regexprep(modName, '[^a-zA-Z0-9]', '_'); % 清理列名
    resultsTable.(['Accuracy_' modNameClean]) = 100 * modAccuracies(mod_idx, :)';
end

% 显示汇总结果
fprintf("\n========== SNR-准确率关系汇总 ==========\n");
disp(resultsTable);

% 保存结果为CSV文件
writetable(resultsTable, fullfile(snrOutputFolder, 'SNR_Accuracy_Results.csv'));

% 6. 分析并输出关键发现
fprintf("\n========== 关键分析结果 ==========\n");
fprintf("1. 最佳准确率: %.2f%% at %d dB\n", 100 * max(accuracies), SNR_list(accuracies == max(accuracies)));
fprintf("2. 最差准确率: %.2f%% at %d dB\n", 100 * min(accuracies), SNR_list(accuracies == min(accuracies)));
fprintf("3. 准确率超过90%%的SNR范围: ");
over_90_idx = find(accuracies >= 0.9);
if ~isempty(over_90_idx)
    fprintf("%d dB 及以上\n", SNR_list(over_90_idx(1)));
else
    fprintf("无 (最高准确率: %.2f%%)\n", 100 * max(accuracies));
end

% 找出在低SNR下表现最好和最差的调制方式
[~, lowSNR_idx] = min(abs(SNR_list - (-5))); % 找到最接近-5dB的点
if lowSNR_idx <= numel(SNR_list)
    [bestAtLowSNR, bestIdx] = max(modAccuracies(:, lowSNR_idx));
    [worstAtLowSNR, worstIdx] = min(modAccuracies(:, lowSNR_idx));
    fprintf("4. 在低SNR(%d dB)下:\n", SNR_list(lowSNR_idx));
    fprintf("   - 表现最好的调制: %s (准确率: %.1f%%)\n", ...
        char(modulationTypes(bestIdx)), 100 * bestAtLowSNR);
    fprintf("   - 表现最差的调制: %s (准确率: %.1f%%)\n", ...
        char(modulationTypes(worstIdx)), 100 * worstAtLowSNR);
end

% 找出在高SNR下仍易混淆的调制对
[~, highSNR_idx] = max(SNR_list); % 最高SNR
if highSNR_idx <= numel(SNR_list) && highSNR_idx <= size(allTestPreds, 2)
    fprintf("5. 在高SNR(%d dB)下的主要混淆:\n", SNR_list(highSNR_idx));
    analyzeConfusionAtSNR(allTestLabels{highSNR_idx}, allTestPreds{highSNR_idx}, modulationTypes);
end

fprintf("\n所有SNR分析结果已保存至: %s\n", snrOutputFolder);

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
%% =========================== 新增的SNR分析函数 ===========================

function [testFrames, testLabels] = generateTestDatasetForSNR(...
    modulationTypes, framesPerModType, sps, spf, fs, SNR, transDelay)
    % 为特定SNR生成测试数据集
    % 输入:
    %   modulationTypes - 调制类型列表
    %   framesPerModType - 每种调制类型的帧数
    %   sps, spf, fs, SNR, transDelay - 信号参数
    % 输出:
    %   testFrames - 测试帧 (1 × spf × 2 × 总帧数)
    %   testLabels - 测试标签
    
    numModTypes = numel(modulationTypes);
    totalFrames = numModTypes * framesPerModType;
    testFrames = zeros(1, spf, 2, totalFrames, 'single');
    testLabels = categorical(repmat(string(modulationTypes(1)), totalFrames, 1));
    testLabels = setcats(testLabels, categories(modulationTypes));
    
    frameCnt = 1;
    for i = 1:numModTypes
        modType = modulationTypes(i);
        
        dataSrc = helperModClassGetSource(string(modType), sps, 2*spf, fs);
        modulator = helperModClassGetModulator(string(modType), sps, fs);
        
        if contains(char(modType), {'B-FM','DSB-AM','SSB-AM'})
            centerFrequency = 100e6;
        else
            centerFrequency = 902e6;
        end
        
        for p = 1:framesPerModType
            x = dataSrc();
            tx = modulator(x);
            rx = applyChannelImpairments(tx, fs, SNR, centerFrequency);
            oneFrame = helperModClassFrameGenerator(rx, spf, spf, transDelay, sps);
            oneFrame = oneFrame(:,1);
            
            testFrames(:,:,:,frameCnt) = single(complexFrameToCNNInput(oneFrame));
            testLabels(frameCnt) = modType;
            
            frameCnt = frameCnt + 1;
        end
    end
end

function analyzeConfusionAtSNR(trueLabels, predLabels, modulationTypes)
    % 分析在特定SNR下的混淆情况
    % 找出最易混淆的调制对
    
    cm = confusionmat(trueLabels, predLabels);
    numClasses = numel(modulationTypes);
    
    % 计算每对调制类型之间的混淆比例
    confusionPairs = [];
    
    for i = 1:numClasses
        for j = 1:numClasses
            if i ~= j && cm(i, j) > 0
                total_i = sum(cm(i, :));
                confusionRate = cm(i, j) / total_i;
                
                confusionPairs(end+1).True = char(modulationTypes(i));
                confusionPairs(end).Predicted = char(modulationTypes(j));
                confusionPairs(end).Count = cm(i, j);
                confusionPairs(end).Rate = confusionRate;
                confusionPairs(end).TotalTrue = total_i;
            end
        end
    end
    
    % 按混淆率排序
    if ~isempty(confusionPairs)
        [~, sortIdx] = sort([confusionPairs.Rate], 'descend');
        confusionPairs = confusionPairs(sortIdx);
        
        % 显示前5个最易混淆的调制对
        fprintf("   最易混淆的调制对:\n");
        for k = 1:min(5, numel(confusionPairs))
            fprintf("   %s → %s: %d/%d (%.1f%%)\n", ...
                confusionPairs(k).True, ...
                confusionPairs(k).Predicted, ...
                confusionPairs(k).Count, ...
                confusionPairs(k).TotalTrue, ...
                100 * confusionPairs(k).Rate);
        end
    else
        fprintf("   无混淆 (100%% 准确率)\n");
    end
end

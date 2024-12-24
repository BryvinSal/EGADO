close all;
clear all;

load("analog_pattern.mat");
load("pixel_pattern.mat");

pixel_size = 100;
mesh_size = 20;

analog_pattern = eps(:,:,6)';
analog_pattern = analog_pattern(1:end-1, 1:end-1);
analog_pattern = flip(analog_pattern, 1);
max_eps = max(analog_pattern(:));
min_eps = min(analog_pattern(:));

analog_pattern = 255 * (analog_pattern - min_eps) / (max_eps - min_eps);

pixel_pattern = pattern*255;

figure
subplot(121)
imshow(analog_pattern);
subplot(122)
imshow(pattern);

% 使用Canny边缘检测算法进行边缘提取
edge_img = edge(analog_pattern, 'Canny');

% 显示结果
figure
subplot(1, 2, 1);
imshow(analog_pattern);
title('原始图像');

subplot(1, 2, 2);
imshow(edge_img);
title('Canny边缘检测结果');

scale = pixel_size/mesh_size;
buff = 4; % must be 2*n, n= 1, 2, 3...
decision_area = scale+buff;


conv_result_1 = convol(edge_img, scale, scale, 0);    % kernel, stride, padding
mapped_result = pattern_mapping(analog_pattern, conv_result_1, 255/2, scale);
error = mapping_error(mapped_result/255, pattern/255);

figure
subplot(131)
imshow(pixel_pattern/255)
subplot(132)
imshow(mapped_result/255)
subplot(133)
imshow(error)

conv_result_2 = convol(edge_img, decision_area, scale, buff/2);    % kernel, stride, padding
mapped_result = pattern_mapping(analog_pattern, conv_result_2, 255/2, scale);
error = mapping_error(mapped_result/255, pattern/255);

% disp("----------------------------------")
% disp("Starting buffer sweep...")
% for buff=0:2:8
%     conv_result = convol(edge_img, scale+buff, scale, buff/2);    % kernel, stride, padding
%     mapped_result = pattern_mapping(analog_pattern, conv_result, 255/2, scale);
%     error = mapping_error(mapped_result/255, pattern/255);
% end

figure
subplot(131)
imshow(pixel_pattern/255)
subplot(132)
imshow(mapped_result/255)
subplot(133)
imshow(error)

% 显示新矩阵
figure
subplot(161)
imshow(analog_pattern/255)
subplot(162)
imshow(pixel_pattern/255)
subplot(163)
imshow(edge_img)
subplot(164)
imshow(conv_result_1);
subplot(165)
imshow(conv_result_2)
subplot(166)
imshow(mapped_result/255)


function result = convol(matrix, kernelSize, stride, padding)
    % 确定矩阵和卷积核的大小
    [m, n] = size(matrix);
    kernel = ones(kernelSize);

    % 计算填充后的矩阵大小
    paddedMatrix = padarray(matrix, [padding, padding], 'both');

    % 计算输出矩阵大小
    outputSize = floor((size(paddedMatrix) - size(kernel)) / stride) + 1;
    result = zeros(outputSize);

    % 执行卷积操作
    for i = 1:stride:(m + 2 * padding - kernelSize + 1)
        for j = 1:stride:(n + 2 * padding - kernelSize + 1)
            window = paddedMatrix(i:i+kernelSize-1, j:j+kernelSize-1);
            result((i-1)/stride+1, (j-1)/stride+1) = logical(sum(sum(window .* kernel)));
        end
    end
    conversion_rate = 100 - (100 * sum(result, 'all')/numel(result));
    disp("Direct Conversion Rate = " + num2str(conversion_rate) + "%");
end

function result_matrix = pattern_mapping(A, B, value, scale)

    % 计算新矩阵的大小
    new_rows = length(A(:,1)) / scale;
    new_cols = length(A(1,:)) / scale;
    
    % 初始化新矩阵
    new_matrix = zeros(new_rows, new_cols);
    
    % 循环遍历原矩阵的每个 mxm 区域，提取中心元素到新矩阵
    for i = 1:new_rows
        for j = 1:new_cols
            % 计算当前 mxm 区域的行和列范围
            row_range = (i - 1) * scale + (1:scale);
            col_range = (j - 1) * scale + (1:scale);
            
            % 计算中心元素的行和列索引
            center_row = floor(mean(row_range));
            center_col = floor(mean(col_range));
            
            % 提取中心元素到新矩阵中对应位置
            new_matrix(i, j) = A(center_row, center_col);
        end
    end
    
    A = new_matrix;
    % 创建一个新的矩阵，将其初始化为与 B 大小相同的零矩阵
    result_matrix = zeros(size(B));

    % 创建逻辑索引，找到 B 中为 0 的元素
    zero_indices = (B == 0);

    % 将 B 中为 0 的位置的元素设为 A 中对应位置的元素
    result_matrix(zero_indices) = A(zero_indices);

    % 创建逻辑索引，找到 B 中为 1 的元素
    one_indices = (B == 1);

    % 将 B 中为 1 的位置的元素设为 255/2
    result_matrix(one_indices) = value;
end

function result_matrix = mapping_error(B, C)
    % 创建一个新的矩阵，将其初始化为与 B 大小相同的零矩阵
    result_matrix = zeros(size(B));

    % 创建逻辑索引，找到 B 中值为 0 或 1 的元素
    zero_or_one_indices = (B == 0) | (B == 1);

    % 比较 B 中为 0 或 1 的元素与 C 中对应位置的元素
    % 如果不相等，将 result_matrix 中对应位置的元素设为 1，否则为 0
    result_matrix(zero_or_one_indices) = (B(zero_or_one_indices) ~= C(zero_or_one_indices));
    error_rate = 100 * sum(result_matrix, 'all') / numel(result_matrix);

    % 将像素错误率显示为字符串形式
    disp("Pixel Error Rate = " + num2str(error_rate) + "%");
end


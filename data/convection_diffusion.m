% Generate the 2D advection--diffusion benchmark data used by FiLM-OSG.
%
% Output files:
%   train_data.mat
%   test_data.mat
%
% Each .mat file contains:
%   coordinates: 64 x 64 x 2 grid coordinates
%   dt:          variable lag intervals
%   trajectories: state snapshots, shape (N, 64, 64, 1, T)
%
% Set make_plots=false below for headless/server runs.

clear; clc; close all;
rng(42);
make_plots = true;

script_path = mfilename('fullpath');
script_dir = fileparts(script_path);
if isempty(script_dir)
    script_dir = pwd;
end

%% Parameters
alpha1 = 1.0;
alpha2 = 0.5;
sigma1 = 0.1;
sigma2 = 0.2;

L = 2 * pi;
N = 64;

n_train_trajectories = 100;
n_test_trajectories = 100;
train_steps = 50;
test_steps = 99;

dt_min = 0.005;
dt_max = 0.5;
n_modes = 25;

fprintf('Advection--diffusion parameters:\n');
fprintf('  advection: (%.1f, %.1f)\n', alpha1, alpha2);
fprintf('  diffusion: (%.1f, %.1f)\n', sigma1, sigma2);

%% Grid and Fourier modes
x = linspace(-pi, pi, N);
y = linspace(-pi, pi, N);
[X, Y] = meshgrid(x, y);

coordinates = zeros(N, N, 2);
coordinates(:, :, 1) = X;
coordinates(:, :, 2) = Y;

kx = (2 * pi / L) * [0:N/2-1, -N/2:-1];
ky = (2 * pi / L) * [0:N/2-1, -N/2:-1];
[KX, KY] = meshgrid(kx, ky);

%% Training set
fprintf('\nGenerating training data (%d trajectories, %d snapshots each)...\n', ...
    n_train_trajectories, train_steps + 1);

dt = zeros(n_train_trajectories, train_steps);
trajectories = zeros(n_train_trajectories, N, N, 1, train_steps + 1);

for traj_idx = 1:n_train_trajectories
    fprintf('  train trajectory %d/%d\n', traj_idx, n_train_trajectories);

    u0 = generate_random_initial_condition(N, n_modes, 1.0);
    trajectories(traj_idx, :, :, 1, 1) = u0;

    dt_seq = dt_min + (dt_max - dt_min) * rand(1, train_steps);
    dt(traj_idx, :) = dt_seq;

    u_current = u0;
    for step = 1:train_steps
        u_current = exact_spectral_evolution( ...
            u_current, dt_seq(step), alpha1, alpha2, sigma1, sigma2, KX, KY);
        trajectories(traj_idx, :, :, 1, step + 1) = u_current;
    end
end

fprintf('Training data shapes:\n');
fprintf('  dt: %s\n', mat2str(size(dt)));
fprintf('  trajectories: %s\n', mat2str(size(trajectories)));
save(fullfile(script_dir, 'train_data.mat'), 'coordinates', 'dt', 'trajectories');

clear dt trajectories;

%% Test set
fprintf('\nGenerating test data (%d trajectories, %d snapshots each)...\n', ...
    n_test_trajectories, test_steps + 1);

dt = zeros(n_test_trajectories, test_steps);
trajectories = zeros(n_test_trajectories, N, N, 1, test_steps + 1);

for traj_idx = 1:n_test_trajectories
    fprintf('  test trajectory %d/%d\n', traj_idx, n_test_trajectories);

    u0 = generate_random_initial_condition(N, n_modes, 1.0);
    trajectories(traj_idx, :, :, 1, 1) = u0;

    dt_seq = dt_min + (dt_max - dt_min) * rand(1, test_steps);
    dt(traj_idx, :) = dt_seq;

    u_current = u0;
    for step = 1:test_steps
        u_current = exact_spectral_evolution( ...
            u_current, dt_seq(step), alpha1, alpha2, sigma1, sigma2, KX, KY);
        trajectories(traj_idx, :, :, 1, step + 1) = u_current;
    end
end

fprintf('Test data shapes:\n');
fprintf('  dt: %s\n', mat2str(size(dt)));
fprintf('  trajectories: %s\n', mat2str(size(trajectories)));
save(fullfile(script_dir, 'test_data.mat'), 'dt', 'trajectories');

%% Verification and diagnostic plots
fprintf('\nVerifying generated files...\n');
train_data = load(fullfile(script_dir, 'train_data.mat'));
test_data = load(fullfile(script_dir, 'test_data.mat'));

fprintf('Training data:\n');
fprintf('  coordinates: %s\n', mat2str(size(train_data.coordinates)));
fprintf('  dt: %s\n', mat2str(size(train_data.dt)));
fprintf('  trajectories: %s\n', mat2str(size(train_data.trajectories)));

fprintf('Test data:\n');
fprintf('  dt: %s\n', mat2str(size(test_data.dt)));
fprintf('  trajectories: %s\n', mat2str(size(test_data.trajectories)));

fprintf('Value ranges:\n');
fprintf('  train: [%.2e, %.2f]\n', ...
    min(train_data.trajectories(:)), max(train_data.trajectories(:)));
fprintf('  test:  [%.2e, %.2f]\n', ...
    min(test_data.trajectories(:)), max(test_data.trajectories(:)));

if make_plots
    plot_diagnostics(train_data, test_data, x, y, N, script_dir);
end

fprintf('\nData generation complete.\n');
fprintf('Output directory: %s\n', script_dir);
fprintf('Generated files:\n');
fprintf('  train_data.mat\n');
fprintf('  test_data.mat\n');
if make_plots
    fprintf('  simplified_visualization.png\n');
    fprintf('  energy_evolution.png\n');
end

%% Local helper functions
function u_new = exact_spectral_evolution(u, dt, alpha1, alpha2, sigma1, sigma2, KX, KY)
    u_hat = fft2(u);
    linear_operator = -1i * (alpha1 .* KX + alpha2 .* KY) ...
        - (sigma1 .* KX.^2 + sigma2 .* KY.^2);
    evolution_factor = exp(linear_operator .* dt);
    u_new = real(ifft2(evolution_factor .* u_hat));
end

function u0 = generate_random_initial_condition(N, n_modes, amplitude_scale)
    u_hat = zeros(N, N);
    k_max = min(floor(sqrt(n_modes)), 5);

    all_modes = [];
    for i = -k_max:k_max
        for j = -k_max:k_max
            if i == 0 && j == 0
                continue;
            end
            k_mag = sqrt(i^2 + j^2);
            if k_mag <= k_max
                all_modes = [all_modes; i, j, k_mag]; %#ok<AGROW>
            end
        end
    end

    [~, idx] = sort(all_modes(:, 3));
    selected_modes = all_modes(idx(1:min(n_modes, length(idx))), :);

    for m = 1:size(selected_modes, 1)
        i = selected_modes(m, 1);
        j = selected_modes(m, 2);
        k_mag = selected_modes(m, 3);

        base_amplitude = amplitude_scale / (1 + k_mag^2);
        amplitude = base_amplitude * (0.5 + 0.5 * rand());
        phase = 2 * pi * rand();

        idx_i = mod(i, N) + 1;
        idx_j = mod(j, N) + 1;
        idx_i_sym = mod(-i, N) + 1;
        idx_j_sym = mod(-j, N) + 1;

        if rand() > 0.5
            u_hat(idx_i, idx_j) = amplitude * exp(1i * phase);
            u_hat(idx_i_sym, idx_j_sym) = amplitude * exp(-1i * phase);
        else
            u_hat(idx_i, idx_j) = 1i * amplitude * exp(1i * phase);
            u_hat(idx_i_sym, idx_j_sym) = -1i * amplitude * exp(-1i * phase);
        end
    end

    u0 = real(ifft2(u_hat));
    current_max = max(abs(u0(:)));
    if current_max > 0
        u0 = u0 / current_max * amplitude_scale;
    else
        u0 = amplitude_scale * randn(N, N) * 0.1;
    end
end

function plot_diagnostics(train_data, test_data, x, y, N, output_dir)
    traj_indices = [1, 2];

    figure('Position', [100, 100, 1200, 400]);
    for i = 1:2
        traj_idx = traj_indices(i);

        subplot(2, 4, (i - 1) * 4 + 1);
        u0 = squeeze(train_data.trajectories(traj_idx, :, :, 1, 1));
        imagesc(x, y, u0);
        colorbar; axis equal tight;
        title(sprintf('Train %d: initial', traj_idx));
        xlabel('x'); ylabel('y');

        subplot(2, 4, (i - 1) * 4 + 2);
        u_final = squeeze(train_data.trajectories(traj_idx, :, :, 1, end));
        imagesc(x, y, u_final);
        colorbar; axis equal tight;
        title(sprintf('Train %d: final', traj_idx));
        xlabel('x'); ylabel('y');

        subplot(2, 4, (i - 1) * 4 + 3);
        u0_test = squeeze(test_data.trajectories(traj_idx, :, :, 1, 1));
        imagesc(x, y, u0_test);
        colorbar; axis equal tight;
        title(sprintf('Test %d: initial', traj_idx));
        xlabel('x'); ylabel('y');

        subplot(2, 4, (i - 1) * 4 + 4);
        u_final_test = squeeze(test_data.trajectories(traj_idx, :, :, 1, end));
        imagesc(x, y, u_final_test);
        colorbar; axis equal tight;
        title(sprintf('Test %d: final', traj_idx));
        xlabel('x'); ylabel('y');
    end
    colormap(jet);
    saveas(gcf, fullfile(output_dir, 'simplified_visualization.png'));

    figure('Position', [100, 100, 800, 600]);
    for i = 1:2
        traj_idx = traj_indices(i);

        subplot(2, 2, (i - 1) * 2 + 1);
        [time_train, energy_train] = energy_curve(train_data, traj_idx, N);
        plot(time_train, energy_train, 'b-', 'LineWidth', 2);
        xlabel('time'); ylabel('mean energy');
        title(sprintf('Train %d: energy', traj_idx));
        grid on;

        subplot(2, 2, (i - 1) * 2 + 2);
        [time_test, energy_test] = energy_curve(test_data, traj_idx, N);
        plot(time_test, energy_test, 'r-', 'LineWidth', 2);
        xlabel('time'); ylabel('mean energy');
        title(sprintf('Test %d: energy', traj_idx));
        grid on;
    end
    saveas(gcf, fullfile(output_dir, 'energy_evolution.png'));
end

function [time_points, energy] = energy_curve(data, traj_idx, N)
    n_steps = size(data.trajectories, 5);
    energy = zeros(n_steps, 1);
    time_points = zeros(n_steps, 1);

    current_time = 0;
    for step = 1:n_steps
        u_current = squeeze(data.trajectories(traj_idx, :, :, 1, step));
        energy(step) = sum(u_current(:).^2) / (N * N);
        time_points(step) = current_time;
        if step < n_steps && step <= size(data.dt, 2)
            current_time = current_time + data.dt(traj_idx, step);
        end
    end
end

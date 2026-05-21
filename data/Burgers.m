% Generate the Burgers benchmark data used by the FiLM-OSG experiments.
%
% Output files:
%   BurgersOSG_train.mat
%   BurgersOSG_test.mat
%
% Each .mat file contains:
%   coordinates: coarse spatial grid, shape (64, 1)
%   dt:          variable lag intervals
%   trajectories: state snapshots, shape (N, 64, 1, T)
%
% Set make_plots=false below for headless/server runs.

clear; close all; clc;
rng(42);
make_plots = true;

script_path = mfilename('fullpath');
script_dir = fileparts(script_path);
if isempty(script_dir)
    script_dir = pwd;
end

%% Parameters
L_fine = 4096;
L_coarse = 64;
D = 1;
N_train = 800;
N_test = 100;
T_train = 11;
T_test = 21;
domain_length = pi;

dt_min = 0.005;
dt_max = 0.15;
CFL = 0.5;
N_modes = 10;

fprintf('Generating Burgers data: fine grid %d -> coarse grid %d\n', ...
    L_fine, L_coarse);

%% Grids
x_fine = linspace(-domain_length, domain_length, L_fine + 1);
x_fine = x_fine(1:end-1)';
dx_fine = x_fine(2) - x_fine(1);

x_coarse = linspace(-domain_length, domain_length, L_coarse + 1);
x_coarse = x_coarse(1:end-1)';
coordinates = x_coarse;

downsample_ratio = L_fine / L_coarse;
downsample_idx = 1:downsample_ratio:L_fine;

%% Training set
fprintf('Generating training data (%d trajectories)...\n', N_train);
tic_total = tic;

trajectories = zeros(N_train, L_coarse, D, T_train, 'single');
dt_train = zeros(N_train, T_train - 1, 'single');

[a0_train, an_train, bn_train] = sample_fourier_coefficients(N_train, N_modes);

for m = 1:N_train
    if mod(m, 200) == 0
        fprintf('  Progress: %d/%d (%.1f min)\n', m, N_train, toc(tic_total) / 60);
    end

    u0_coarse = initial_condition(x_coarse, a0_train(m), an_train(m, :), bn_train(m, :));
    u0_fine = interp1(x_coarse, u0_coarse, x_fine, 'spline');

    time_steps = dt_min + (dt_max - dt_min) * rand(T_train - 1, 1);
    dt_train(m, :) = time_steps';
    t_points = [0; cumsum(time_steps)];

    u_current = u0_fine;
    trajectories(m, :, 1, 1) = u0_coarse;

    for s = 1:T_train - 1
        target_time = t_points(s + 1);
        current_time = t_points(s);

        while current_time < target_time
            dt_adaptive = min(dt_max, target_time - current_time);
            max_u = max(abs(u_current));
            dt_cfl = CFL * dx_fine / (max_u + eps);
            dt_use = min(dt_adaptive, dt_cfl);

            u_current = RK4_LF_step(u_current, dt_use, dx_fine);
            current_time = current_time + dt_use;
        end

        trajectories(m, :, 1, s + 1) = u_current(downsample_idx);
    end
end

dt = dt_train;
save(fullfile(script_dir, 'BurgersOSG_train.mat'), ...
    'coordinates', 'dt', 'trajectories', '-v7');
fprintf('Training data saved.\n');

%% Test set
fprintf('\nGenerating test data (%d trajectories)...\n', N_test);

trajectories = zeros(N_test, L_coarse, D, T_test, 'single');
dt_test = zeros(N_test, T_test - 1, 'single');

[a0_test, an_test, bn_test] = sample_fourier_coefficients(N_test, N_modes);

for m = 1:N_test
    if mod(m, 50) == 0
        fprintf('  Progress: %d/%d\n', m, N_test);
    end

    if m == N_test
        u0_coarse = -sin(x_coarse);
        total_time = 2.0;
        time_steps = (total_time / (T_test - 1)) * ones(T_test - 1, 1);
        fprintf('  Test case %d: shock wave initial condition -sin(x)\n', m);
    else
        u0_coarse = initial_condition(x_coarse, a0_test(m), an_test(m, :), bn_test(m, :));
        time_steps = dt_min + (dt_max - dt_min) * rand(T_test - 1, 1);
    end

    dt_test(m, :) = time_steps';
    u0_fine = interp1(x_coarse, u0_coarse, x_fine, 'spline');
    t_points = [0; cumsum(time_steps)];

    u_current = u0_fine;
    trajectories(m, :, 1, 1) = u0_coarse;

    for s = 1:T_test - 1
        target_time = t_points(s + 1);
        current_time = t_points(s);

        while current_time < target_time
            dt_adaptive = min(dt_max, target_time - current_time);
            max_u = max(abs(u_current));
            dt_cfl = CFL * dx_fine / (max_u + eps);
            dt_use = min(dt_adaptive, dt_cfl);

            u_current = RK4_LF_step(u_current, dt_use, dx_fine);
            current_time = current_time + dt_use;
        end

        trajectories(m, :, 1, s + 1) = u_current(downsample_idx);
    end
end

dt = dt_test;
save(fullfile(script_dir, 'BurgersOSG_test.mat'), ...
    'coordinates', 'dt', 'trajectories', '-v7');
fprintf('Test data saved.\n');

%% Optional diagnostic plot for the shock test case
if make_plots
    t_last_case = [0; cumsum(dt_test(end, :)')];
    plot_waveforms_at_times( ...
        trajectories, ...
        x_coarse, ...
        t_last_case, ...
        fullfile(script_dir, 'BurgersOSG_Shock_Waveforms.png') ...
    );
end

%% Local helper functions
function [a0, an, bn] = sample_fourier_coefficients(N, N_modes)
    a0 = -0.5 + rand(N, 1);
    an = zeros(N, N_modes);
    bn = zeros(N, N_modes);
    for n = 1:N_modes
        an(:, n) = -1 / n + (2 / n) * rand(N, 1);
        bn(:, n) = -1 / n + (2 / n) * rand(N, 1);
    end
end

function u0 = initial_condition(x, a0, an, bn)
    u0 = a0 * ones(size(x));
    for n = 1:numel(an)
        u0 = u0 + an(n) * cos(n * x) + bn(n) * sin(n * x);
    end
end

function dudt = burgers_rhs_LF(u, dx)
    f = 0.5 * u.^2;
    u_p = circshift(u, -1);
    u_m = circshift(u, 1);
    f_p = circshift(f, -1);
    f_m = circshift(f, 1);
    alpha = max(abs(u)) + eps;
    F_plus = 0.5 * (f + f_p) - 0.5 * alpha * (u_p - u);
    F_minus = 0.5 * (f_m + f) - 0.5 * alpha * (u - u_m);
    dudt = -(F_plus - F_minus) / dx;
end

function u_next = RK4_LF_step(u, dt, dx)
    k1 = burgers_rhs_LF(u, dx);
    k2 = burgers_rhs_LF(u + 0.5 * dt * k1, dx);
    k3 = burgers_rhs_LF(u + 0.5 * dt * k2, dx);
    k4 = burgers_rhs_LF(u + dt * k3, dx);
    u_next = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    u_next = max(min(u_next, 10), -10);
end

function plot_waveforms_at_times(trajectories, x, t_points, save_filename)
    u_traj = squeeze(trajectories(end, :, 1, :));
    n_times = length(t_points);
    indices = unique([1, round(linspace(2, n_times, 5))]);
    if indices(end) ~= n_times
        indices = [indices, n_times];
    end
    plot_times = t_points(indices);
    colors = lines(length(indices));

    figure('Position', [100, 100, 1200, 700], 'Color', 'w');
    for i = 1:length(indices)
        subplot(2, 3, i);
        idx = indices(i);
        plot(x, u_traj(:, idx), 'LineWidth', 2, 'Color', colors(i, :));
        hold on; grid on;
        title(sprintf('t = %.2f', plot_times(i)));
        xlabel('x'); ylabel('u(x)'); ylim([-2.2, 2.2]);
        if plot_times(i) > 0.5
            [~, max_idx] = max(abs(gradient(u_traj(:, idx), x(2) - x(1))));
            plot(x(max_idx), u_traj(max_idx, idx), ...
                'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        end
    end

    subplot(2, 3, [4 6]);
    for i = 1:length(indices)
        idx = indices(i);
        plot(x, u_traj(:, idx), ...
            'LineWidth', 1.5, ...
            'Color', colors(i, :), ...
            'DisplayName', sprintf('t = %.2f', plot_times(i)));
        hold on;
    end
    grid on; xlabel('x'); ylabel('u(x)'); title('Time Evolution Overlay');
    legend('Location', 'best'); ylim([-2.2, 2.2]);
    sgtitle('Burgers Equation Shock Formation (-sin(x))', ...
        'FontSize', 14, 'FontWeight', 'bold');

    if ~isempty(save_filename)
        saveas(gcf, save_filename);
        fprintf('Plot saved: %s\n', save_filename);
    end
    drawnow;
end

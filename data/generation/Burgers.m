% Generate the Burgers benchmark data used by the FiLM-OSG experiments.
%
% Output files:
%   BurgersOSG_train.mat
%   BurgersOSG_test.mat
%
% Each .mat file contains:
%   coordinates: coarse spatial grid, shape (64, 1)
%   dt:          varying time intervals
%   trajectories: state snapshots, shape (N, 64, 1, T)
%
clear; clc;
rng(42);

script_path = mfilename('fullpath');
script_dir = fileparts(script_path);
if isempty(script_dir)
    script_dir = pwd;
end

output_dir = getenv('FILM_OSG_OUTPUT_DIR');
if isempty(output_dir)
    output_dir = fileparts(script_dir);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
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
    u0_fine = initial_condition(x_fine, a0_train(m), an_train(m, :), bn_train(m, :));

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
train_output = fullfile(output_dir, 'BurgersOSG_train.mat');
assert(~exist(train_output, 'file'), ...
    'Refusing to overwrite existing file: %s', train_output);
save(train_output, ...
    'coordinates', 'dt', 'trajectories', '-v7');
assert_shape(size(coordinates), [L_coarse, 1], 'train coordinates');
assert_shape(size(dt), [N_train, T_train - 1], 'train dt');
assert_shape(size(trajectories), [N_train, L_coarse, D, T_train], ...
    'train trajectories');
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
        u0_fine = -sin(x_fine);
        total_time = 2.0;
        time_steps = (total_time / (T_test - 1)) * ones(T_test - 1, 1);
        fprintf('  Test case %d: shock wave initial condition -sin(x)\n', m);
    else
        u0_coarse = initial_condition(x_coarse, a0_test(m), an_test(m, :), bn_test(m, :));
        u0_fine = initial_condition(x_fine, a0_test(m), an_test(m, :), bn_test(m, :));
        time_steps = dt_min + (dt_max - dt_min) * rand(T_test - 1, 1);
    end

    dt_test(m, :) = time_steps';
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
test_output = fullfile(output_dir, 'BurgersOSG_test.mat');
assert(~exist(test_output, 'file'), ...
    'Refusing to overwrite existing file: %s', test_output);
save(test_output, ...
    'coordinates', 'dt', 'trajectories', '-v7');
assert_shape(size(coordinates), [L_coarse, 1], 'test coordinates');
assert_shape(size(dt), [N_test, T_test - 1], 'test dt');
assert_shape(size(trajectories), [N_test, L_coarse, D, T_test], ...
    'test trajectories');
fprintf('Test data saved.\n');

fprintf('Burgers data generation complete.\n');

%% Local helper functions
function assert_shape(actual, expected, name)
    assert(isequal(actual, expected), ...
        '%s shape mismatch. Expected %s, got %s.', ...
        name, mat2str(expected), mat2str(actual));
    fprintf('Verified %s shape: %s\n', name, mat2str(actual));
end

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
    assert(all(isfinite(u_next)), 'Non-finite value produced by Burgers solver.');
    assert(max(abs(u_next)) < 10, ...
        'Burgers solver exceeded the safety bound |u| < 10.');
end

% Generate advection--diffusion extrapolation sets at fixed evolution times.
%
% This diagnostic is separate from the main advection--diffusion benchmark.
% The main benchmark uses train/test time intervals in [0.005, 0.5]. Here we
% generate fixed-time test sets outside that interval to probe
% extrapolation behavior without changing the main data files.
%
% Output files:
%   test_data_fixed_dt_0p0025.mat
%   test_data_fixed_dt_0p75.mat
%   test_data_fixed_dt_1.mat
%
% Each output contains:
%   coordinates:   64 x 64 x 2 grid coordinates
%   dt:            100 x 20 fixed time intervals
%   trajectories:  100 x 64 x 64 x 1 x 21 snapshots
%
clear; clc;

fixed_time_seed = 100042;
rng(fixed_time_seed);

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

%% Parameters copied from the main advection--diffusion generator.
alpha1 = 1.0;
alpha2 = 0.5;
sigma1 = 0.1;
sigma2 = 0.2;

L = 2 * pi;
N = 64;
n_test_trajectories = 100;
test_steps = 20;
n_modes = 25;

train_dt_min = 0.005;
train_dt_max = 0.5;
fixed_dt_values = [0.0025, 0.75, 1.0];

fprintf('Advection--diffusion fixed-time extrapolation diagnostic:\n');
fprintf('  training time interval: [%.4g, %.4g]\n', train_dt_min, train_dt_max);
fprintf('  fixed values of dt: %s\n', mat2str(fixed_dt_values));
fprintf('  trajectories: %d, rollout steps per dt: %d\n', ...
    n_test_trajectories, test_steps);
fprintf('  random seed: %d\n', fixed_time_seed);
fprintf('  output directory: %s\n', output_dir);

%% Periodic grid and Fourier modes.
dx = L / N;
x = -pi + (0:N-1) * dx;
y = -pi + (0:N-1) * dx;
[X, Y] = meshgrid(x, y);

coordinates = zeros(N, N, 2);
coordinates(:, :, 1) = X;
coordinates(:, :, 2) = Y;

kx = (2 * pi / L) * [0:N/2-1, -N/2:-1];
ky = (2 * pi / L) * [0:N/2-1, -N/2:-1];
[KX, KY] = meshgrid(kx, ky);
assert(abs(x(2) - x(1) - dx) < 10 * eps(dx), ...
    'Periodic grid spacing mismatch.');

%% Shared initial conditions for all fixed-time test sets.
initial_conditions = zeros(n_test_trajectories, N, N);
for traj_idx = 1:n_test_trajectories
    initial_conditions(traj_idx, :, :) = ...
        generate_random_initial_condition(N, n_modes, 1.0);
end

%% Generate one file per fixed value of dt.
for dt_idx = 1:numel(fixed_dt_values)
    fixed_dt = fixed_dt_values(dt_idx);
    dt = fixed_dt * ones(n_test_trajectories, test_steps);
    trajectories = zeros(n_test_trajectories, N, N, 1, test_steps + 1);

    fprintf('\nGenerating fixed dt = %.6g\n', fixed_dt);
    for traj_idx = 1:n_test_trajectories
        if mod(traj_idx, 10) == 0 || traj_idx == 1
            fprintf('  trajectory %d/%d\n', traj_idx, n_test_trajectories);
        end

        u_current = squeeze(initial_conditions(traj_idx, :, :));
        trajectories(traj_idx, :, :, 1, 1) = u_current;

        for step = 1:test_steps
            u_current = exact_spectral_evolution( ...
                u_current, fixed_dt, alpha1, alpha2, sigma1, sigma2, KX, KY);
            trajectories(traj_idx, :, :, 1, step + 1) = u_current;
        end
    end

    out_name = sprintf('test_data_fixed_dt_%s.mat', dt_to_token(fixed_dt));
    out_path = fullfile(output_dir, out_name);
    generator_seed = fixed_time_seed;
    save(out_path, 'coordinates', 'dt', 'trajectories', 'generator_seed', '-v7');

    assert_shape(size(coordinates), [N, N, 2], 'coordinates');
    assert_shape(size(dt), [n_test_trajectories, test_steps], 'dt');
    assert_shape(size(trajectories), ...
        [n_test_trajectories, N, N, 1, test_steps + 1], ...
        'trajectories');
    fprintf('Saved %s\n', out_path);
end

fprintf('\nFixed-time extrapolation data generation complete.\n');

%% Local helper functions
function token = dt_to_token(value)
    token = strrep(sprintf('%.6g', value), '.', 'p');
    token = strrep(token, '-', 'm');
end

function assert_shape(actual, expected, name)
    assert(isequal(actual, expected), ...
        '%s shape mismatch. Expected %s, got %s.', ...
        name, mat2str(expected), mat2str(actual));
    fprintf('Verified %s shape: %s\n', name, mat2str(actual));
end

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

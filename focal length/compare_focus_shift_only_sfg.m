function compare_focus_shift_only_sfg()
%% ================ User Inputs =================
% Experimental data (pump wavelength in nm) and focal length (mm)
wpump_exp_nm = [800 825 850 875 900 925 950 975 1000 1025 1050 1075 1100 1125 1150 1175 1200];
f_exp        = [12.2 12   11.6 11.3 11.0 10.7 10.5 10.4 10.3 10.1  9.9  9.8  9.7  9.7  9.6  9.4  9.3];

% Fixed signal wavelength (nm)
signal_fixed_nm = 1550;

% Fresnel zone lens parameters
r1_m = (120*4.5*3/15)*1e-6;   % effective radius r1, e.g., 108 um -> m
n    = 2;                     % diffraction order

% Shift-only on SFG axis (nm)
delta_sfg_bounds_nm = [-40 40];

% Loss type: 'rmse' or 'huber'
loss_type = 'huber';
huber_c   = 0.2;              % in mm

%% ============= Model: SFG + FZL (mm) ==============
sfg_from_pump = @(wp_nm) 1 ./ (1./signal_fixed_nm + 1./wp_nm);
f_th_from_sfg_mm = @(lambda_sfg_nm) (r1_m^2 / n) ./ (lambda_sfg_nm * 1e-9) * 1e3;

lambda_sfg_nom_nm = sfg_from_pump(wpump_exp_nm(:));

%% =============== Optimize Δ_sfg ====================
obj = @(delta_sfg) objective_shift_on_sfg( ...
    delta_sfg, lambda_sfg_nom_nm, f_exp(:), f_th_from_sfg_mm, loss_type, huber_c);

delta_sfg_opt = fminbnd(@(d) obj(d), delta_sfg_bounds_nm(1), delta_sfg_bounds_nm(2));

% Apply optimal Δ_sfg (no scaling, no intercept)
lambda_sfg_used_nm = lambda_sfg_nom_nm + delta_sfg_opt;
f_pred = f_th_from_sfg_mm(lambda_sfg_used_nm);   % mm

%% ================== Metrics ====================
res  = f_exp(:) - f_pred(:);
RMSE = sqrt(mean(res.^2));
MAE  = mean(abs(res));
MAPE = mean(abs(res ./ f_exp(:))) * 100;
SS_res = sum(res.^2);
SS_tot = sum((f_exp(:) - mean(f_exp(:))).^2);
R2 = 1 - SS_res/SS_tot;

%% =================== Plots =====================
% One figure, 800x300 pixels (same size as your single-image example)
fig = figure('Color','w','Position',[100 100 700 300]);

% Use English font across the figure
set(fig, 'DefaultAxesFontName','Arial', 'DefaultTextFontName','Arial');

tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% (1) Curve comparison (x: pump, theory internally uses SFG+shift)
nexttile;
plot(wpump_exp_nm, f_exp, 'o', 'MarkerSize', 6, 'DisplayName','Experiment'); hold on;
wp_dense = linspace(min(wpump_exp_nm), max(wpump_exp_nm), 500);
lambda_sfg_dense = sfg_from_pump(wp_dense) + delta_sfg_opt;
lambda_sfg_dense(lambda_sfg_dense <= 0) = NaN;   % guard
f_dense = f_th_from_sfg_mm(lambda_sfg_dense);
plot(wp_dense, f_dense, '-', 'LineWidth', 1.6, 'DisplayName','Theory (SFG shift only)');
grid on;
xlabel('Pump wavelength (nm)');
ylabel('Focal length (mm)');
title(sprintf('SFG-axis shift \\Delta_{SFG} = %.2f nm', delta_sfg_opt));
legend('Location','best');

% (2) Residuals vs pump
nexttile;
stem(wpump_exp_nm, res, 'filled'); grid on;
xlabel('Pump wavelength (nm)');
ylabel('Residual = Experiment - Theory (mm)');
title('Residuals');

% Tight axes/appearance
set(findall(fig,'-property','FontName'),'FontName','Arial');

%% ============== Summary Print =================
fprintf('=== Shift-only (SFG axis) ===\n');
fprintf('Δ_SFG           = %.6g nm\n', delta_sfg_opt);
fprintf('Loss Type       = %s (Huber c=%.3g)\n', loss_type, huber_c);
fprintf('RMSE            = %.6g mm\n', RMSE);
fprintf('MAE             = %.6g mm\n', MAE);
fprintf('MAPE            = %.3f %%\n', MAPE);
fprintf('R^2             = %.4f\n', R2);
% Optional export:
% T = table(wpump_exp_nm(:), lambda_sfg_nom_nm(:), lambda_sfg_used_nm(:), f_exp(:), f_pred(:), res(:), ...
%     'VariableNames', {'pump_nm','sfg_nom_nm','sfg_used_nm','f_exp_mm','f_theory_mm','residual_mm'});
% writetable(T, 'focus_compare_shift_only_sfg.csv');
end

%% =========== helper: objective on SFG axis ===========
function val = objective_shift_on_sfg(delta_sfg, lambda_sfg_nom_nm, f_exp, f_th_from_sfg_mm, loss_type, huber_c)
    % Ensure column vectors
    lambda_sfg_nom_nm = lambda_sfg_nom_nm(:);
    f_exp             = f_exp(:);

    % Apply SFG shift
    lambda_sfg = lambda_sfg_nom_nm + delta_sfg;

    % Guards
    if any(lambda_sfg <= 0) || any(~isfinite(lambda_sfg))
        val = inf; return;
    end

    % Theory
    f_th = f_th_from_sfg_mm(lambda_sfg);
    if any(~isfinite(f_th)) || numel(f_th) ~= numel(f_exp)
        val = inf; return;
    end

    % Residuals
    r = f_exp - f_th;

    switch lower(loss_type)
        case 'rmse'
            val = sqrt(mean(r.^2));
        case 'huber'
            a = abs(r);
            L = zeros(size(r));
            quad = a <= huber_c;
            L(quad)  = 0.5 * (r(quad)).^2;
            L(~quad) = huber_c * a(~quad) - 0.5 * huber_c^2;
            val = mean(L);
        otherwise
            error('Unknown loss_type: %s', loss_type);
    end
end




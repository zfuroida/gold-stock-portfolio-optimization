% 1. Input data berdasarkan excel (5 emiten emas)
n = 5;
% Expected Return
mu = [0.000868385; 0.001039308; 0.000622992; 0.001644303; 0.002389144];
% Matriks Varians-Kovarians 5x5 
Sigma = [
    0.000195299262172985, 0.0000393410194037496, 0.000108694496595146, 0.0000185316988073998, 0.000130632185167548;
    0.0000393410194037496, 0.0000630610998014361, 0.000016076363995403, 0.0000138535462109159, -0.0000129710997951922;
    0.000108694496595146, 0.000016076363995403, 0.000257361198226241, 0.0000342932988188925, 0.00011058624510302;
    0.0000185316988073998, 0.0000138535462109159, 0.0000342932988188925, 0.000115429487413093, -5.84845679719709E-07;
    0.000130632185167548, -0.0000129710997951922, 0.00011058624510302, -5.84845679719709E-07, 0.000451332423898397
];
% Vektor e untuk batasan total bobot (e' * x = 1)
e = ones(n, 1);
% rho adalah batasan maksimum untuk varians portofolio (x' * Sigma * x)
rho_values = [0.001, 0.0001, 0.00001]; 
disp(['Expected Return (mu):']); disp(mu);
disp(['Matriks Varians-Kovarians (Sigma):']); disp(Sigma);
disp(['Nilai Batasan Risiko (rho) yang digunakan:']); disp(rho_values);

% 2. Pemodelan Masalah (Quadratic Programming)
H = zeros(n, n); % Tidak ada bagian kuadratik (1/2 * x' * H * x = 0)
f = -mu;         % Bagian linier (f' * x = -mu' * x). negatif karena minimize

% 3. Batasan Persamaan (Aeq * x = beq)
% Total bobot investasi: e' * x = 1
Aeq = e'; 
beq = 1;

% 4. Batasan Ketidaksamaan (A * x <= b)
lb = zeros(n, 1); % Batasan bawah (lower bound)
ub = [];          % Batasan atas (upper bound) - tidak ada

% 5. Pemecahan Masalah menggunakan fmincon
% fmincon digunakan karena batasan risiko (x' * Sigma * x <= rho) adalah non-linier.
results = struct('rho', {}, 'weights', {}, 'return', {}, 'risk', {});

for i = 1:length(rho_values)
    rho = rho_values(i);
    % Fungsi Objektif: min f(x) = -mu' * x
    fun = @(x) -mu' * x;
    % Batasan Non-Linier
    nonlcon = @(x) risk_constraint(x, Sigma, rho);
    % Kondisi awal
    x0 = e / n;   
    % Opsi Solver
    options = optimoptions('fmincon', 'Display', 'off');
    % Panggil Solver (fmincon)
    [x_opt, fval] = fmincon(fun, x0, [], [], Aeq, beq, lb, ub, nonlcon, options);
    % Hitung hasil
    return_opt = -fval; % Expected Return = -fval 
    risk_opt = x_opt' * Sigma * x_opt; % Varians Portofolio
    % Simpan hasil
    results(i).rho = rho;
    results(i).weights = x_opt;
    results(i).return = return_opt;
    results(i).risk = risk_opt;
    
end

% 6. Fungsi Batasan Non-Linier
function [c, ceq] = risk_constraint(x, Sigma, rho)
    % Batasan Non-Linier: c(x) <= 0
    c = x' * Sigma * x - rho; 
    
    % Tidak ada batasan persamaan non-linier
    ceq = []; 
end

% 7. Output
disp(' ');
disp('==================================================');
disp('Hasil Optimasi Portofolio (3 Portofolio Berbeda)');
disp('==================================================');

for i = 1:length(results)
    disp(['PORTOFOLIO KE-' num2str(i) ' (Batasan Risiko, rho <= ' num2str(results(i).rho) '):']);
    disp(['  Bobot Optimal (x):']);
    fprintf('    ANTM: %.10f\n', results(i).weights(1));
    fprintf('    UNTR: %.10f\n', results(i).weights(2));
    fprintf('    MDKA: %.10f\n', results(i).weights(3));
    fprintf('    MEDC: %.10f\n', results(i).weights(4));
    fprintf('    PSAB: %.10f\n', results(i).weights(5));
    fprintf('  Total Bobot: %.10f\n', sum(results(i).weights));
    disp(['  Expected Return Portofolio (R_p): ' num2str(results(i).return, '%.10f')]);
    disp(['  Varians Portofolio (x''*Sigma*x): ' num2str(results(i).risk, '%.10f')]);
    disp('--------------------------------------------------');
end

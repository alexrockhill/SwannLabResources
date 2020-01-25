data {
    int<lower=0>            ntimes;
    int<lower=0>            nfreqs;
    real<lower=0>           data[nfreqs, ntimes];   // Morlet tranformed EEG data from a single channel
    real<lower=0>           lower_bound;            // Lower bound to be set to 3x median power in beta range
    real<lower=0>           mu_power_prior;         // Power mean prior to be set to 6x median power in beta range
    real<lower=0>           sigma_t_prior;          // spread in time prior to be set to 50 ms (sfreq * 0.05)
    real<lower=0>           mu_freq_prior;          // middle of the frequency range
    real<lower=0>           sigma_freq_prior;       // std of frequency (about)
}
parameters {
    vector[ntimes]          bursting;
    real<lower=0>           lambda;
    real<lower=0>           sigma_t;
    real<lower=lower_bound> mu_power;
    real<lower=0>           sigma_power;
    real<lower=0>           mu_freq;
    real<lower=0>           sigma_freq;
}
model {
    
    data ~ multi_normal([mu_freq, poisson(lambda)], [sigma_freq, sigma_t]);
    // Note: the means are also used for standard devations because sigma_t
    // could be 50 samples and choosing std = 1 or 10 would not be the best
    // because this changes with sfreq. For power, the mean power could be on the
    // order of 1e-7 so again a std on the prior of 1 or 10 would not make sense.
    lambda ~ gamma(2, 2);  // long tailed prior, could be waiting a while
    sigma_t ~ normal(sigma_t_prior, sigma_t_prior); // mean also used for std so on same scale roughly
    mu_power ~ normal(mu_power_prior, mu_power_prior);
    sigma_power ~ normal(mu_power_prior, mu_power_prior);  // want on same scale as mu_power
    mu_freq ~ normal(mu_freq_prior, sigma_freq_prior);
    sigma_freq ~ normal(sigma_freq_prior, 10);
}
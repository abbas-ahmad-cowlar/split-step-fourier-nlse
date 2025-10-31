"""
FFT Smoke Test for SSFM Project
Verifies NumPy FFT is working correctly — this is the core tool.
"""
import numpy as np


def test_smoke():

    # Test 1: FFT of a simple signal
    N = 1024
    dt = 0.01
    t = np.arange(N) * dt
    signal = np.cos(2 * np.pi * 10 * t)  # 10 Hz cosine
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=dt)

    # The peak should be at ±10 Hz
    peak_freq = freqs[np.argmax(np.abs(spectrum[:N//2]))]
    assert abs(peak_freq - 10.0) < 0.5, f"FFT peak at {peak_freq} Hz, expected 10 Hz"

    # Test 2: Parseval's theorem — energy in time = energy in frequency
    E_time = np.sum(np.abs(signal)**2) * dt
    E_freq = np.sum(np.abs(spectrum)**2) / N * dt  # Parseval normalization
    assert abs(E_time - E_freq) / E_time < 1e-10, "Parseval's theorem violated"

    # Test 3: FFT → IFFT roundtrip
    reconstructed = np.fft.ifft(np.fft.fft(signal))
    assert np.max(np.abs(reconstructed - signal)) < 1e-12, "FFT-IFFT roundtrip failed"

    # Test 4: fftshift moves zero-frequency to center
    shifted_freqs = np.fft.fftshift(freqs)
    assert shifted_freqs[N//2] == 0.0, "fftshift center should be zero frequency"

    # Test 5: sech pulse energy (key SSFM validation)
    tau = np.linspace(-20, 20, 2048, endpoint=False)
    dtau = tau[1] - tau[0]
    abs_tau = np.abs(tau)
    sech = 2 * np.exp(-abs_tau) / (1 + np.exp(-2 * abs_tau))
    energy = np.sum(np.abs(sech)**2) * dtau
    assert abs(energy - 2.0) < 0.01, f"sech energy = {energy}, expected 2.0"

    print("[PASS] All FFT smoke tests passed. NumPy FFT is working correctly.")
    print(f"   int|sech(tau)|^2 dtau = {energy:.6f} (expected: 2.0)")

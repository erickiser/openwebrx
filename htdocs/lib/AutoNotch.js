//
// Automatic Notch Filter
//
// Finds steady single tones (carriers, heterodynes, "birdies") in the
// received audio and removes them with narrow notch filters, leaving
// speech and other changing sounds alone. The filter runs in the
// browser, on audio samples as they arrive from the server, so it
// only affects what this one listener hears and records. Server-side
// decoders are not affected.
//
// How it works:
// 1. About 20 times per second, the detector computes the spectrum
//    of the most recent audio and averages it over time.
// 2. A frequency that sticks out as a sharp, narrow peak above its
//    surroundings, and stays in place for about half a second, is
//    considered a tone. Speech does not do that: its pitch keeps
//    moving and its peaks are much wider.
// 3. Each tone found gets its own notch filter, tuned to the tone's
//    frequency. The notch follows the tone if it drifts, and goes
//    away shortly after the tone disappears.
//

function AutoNotch(sampleRate) {
    this.sampleRate = sampleRate;

    // Spectrum size: a power of two giving frequency resolution
    // of about 6..12Hz per bin
    this.fftSize = 256;
    while (this.fftSize * 12 < sampleRate) this.fftSize *= 2;
    this.binHz   = sampleRate / this.fftSize;
    this.hop     = this.fftSize / 2;

    // Detector settings
    this.maxNotches  = 6;    // maximum number of tones removed at once
    this.threshold   = 31.6; // tone must be 15dB above its surroundings
    this.engageHops  = Math.ceil(0.5 * sampleRate / this.hop); // tone must last 0.5s
    this.releaseHops = Math.ceil(2.0 * sampleRate / this.hop); // notch stays up to 2s after tone ends
    this.steadiness  = 0.35; // new tone must stay within this many bins
    this.averaging   = 0.35; // weight of the newest spectrum in the average
    this.neighbors   = Math.round(150 / this.binHz); // surroundings are +-150Hz
    this.minBin      = Math.ceil(40 / this.binHz);   // ignore everything below 40Hz
    this.notchWidth  = 30;   // notch bandwidth in Hz

    // Detector state
    this.ring    = new Float32Array(this.fftSize); // most recent input samples
    this.ringPos = 0;
    this.pending = 0;                              // samples since last analysis
    this.avg     = new Float32Array(this.fftSize / 2);
    this.re      = new Float32Array(this.fftSize);
    this.im      = new Float32Array(this.fftSize);
    this.tracks  = [];                             // tones being watched or notched

    // Precomputed tables for the FFT and the Hann window
    this.window = new Float32Array(this.fftSize);
    this.cos    = new Float32Array(this.fftSize / 2);
    this.sin    = new Float32Array(this.fftSize / 2);
    this.rev    = new Uint32Array(this.fftSize);
    var bits = Math.round(Math.log2(this.fftSize));
    for (var i = 0; i < this.fftSize; i++) {
        this.window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.fftSize);
        var r = 0;
        for (var b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        this.rev[i] = r;
    }
    for (var i = 0; i < this.fftSize / 2; i++) {
        this.cos[i] = Math.cos(2 * Math.PI * i / this.fftSize);
        this.sin[i] = Math.sin(2 * Math.PI * i / this.fftSize);
    }
}

// Forget all tones found so far.
AutoNotch.prototype.reset = function() {
    this.ring.fill(0);
    this.avg.fill(0);
    this.tracks  = [];
    this.pending = 0;
};

// Get frequencies (in Hz) of the tones currently being removed.
AutoNotch.prototype.getNotches = function() {
    return this.tracks
        .filter(function(t) { return t.active; })
        .map(function(t) { return t.freq; });
};

// Filter audio. Takes an array of Int16Array channels (one for mono,
// two for stereo), returns a new array of the same shape.
AutoNotch.prototype.process = function(channels) {
    var input = channels[0];
    var len   = input.length;
    var out   = channels.map(function(x) { return new Float64Array(x); });

    // Go through the audio in pieces, stopping to look for tones
    // every <hop> samples, so that the notches are always up to date
    var done = 0;
    while (done < len) {
        var count = Math.min(len - done, this.hop - this.pending);

        // Feed the detector (first channel only)
        for (var i = done; i < done + count; i++) {
            this.ring[this.ringPos] = input[i];
            this.ringPos = (this.ringPos + 1) % this.fftSize;
        }

        // Run every active notch over this piece of audio
        for (var t = 0; t < this.tracks.length; t++) {
            var track = this.tracks[t];
            if (!track.active) continue;
            for (var c = 0; c < out.length; c++) {
                this.runNotch(track, c, out[c], done, done + count);
            }
        }

        done += count;
        this.pending += count;
        if (this.pending >= this.hop) {
            this.pending = 0;
            this.analyze();
        }
    }

    // Convert back to 16bit samples
    return out.map(function(x) {
        var y = new Int16Array(x.length);
        for (var i = 0; i < x.length; i++) {
            y[i] = x[i] > 32767? 32767 : x[i] < -32768? -32768 : Math.round(x[i]);
        }
        return y;
    });
};

// Apply one tone's notch filter to samples [from, to) of a channel.
// The filter is two identical second-order sections in a row, which
// makes the notch deeper than a single section would.
AutoNotch.prototype.runNotch = function(track, channel, data, from, to) {
    // Each channel keeps its own filter memory
    var s = track.state[channel];
    if (!s) s = track.state[channel] = new Float64Array(4);

    var b0 = track.b0, b1 = track.b1, a2 = track.a2;
    var s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3];

    for (var i = from; i < to; i++) {
        var x = data[i];
        var y = b0 * x + s0;
        s0 = b1 * (x - y) + s1;
        s1 = b0 * x - a2 * y;
        var z = b0 * y + s2;
        s2 = b1 * (y - z) + s3;
        s3 = b0 * y - a2 * z;
        data[i] = z;
    }

    // Protect against numerical trouble
    if (isNaN(s0 + s1 + s2 + s3)) s0 = s1 = s2 = s3 = 0;

    s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
};

// Compute notch filter coefficients for a tone at given frequency.
AutoNotch.prototype.tuneNotch = function(track) {
    var k = Math.tan(Math.PI * this.notchWidth / this.sampleRate);
    track.b0 = 1 / (1 + k);
    track.b1 = -2 * Math.cos(2 * Math.PI * track.freq / this.sampleRate) / (1 + k);
    track.a2 = (1 - k) / (1 + k);
    track.tuned = track.freq;
};

// Look at the latest audio and update the list of tones.
AutoNotch.prototype.analyze = function() {
    var size = this.fftSize;
    var half = size / 2;
    var re   = this.re;
    var im   = this.im;
    var avg  = this.avg;

    // Windowed copy of the latest samples, in bit-reversed order
    for (var i = 0; i < size; i++) {
        re[this.rev[i]] = this.ring[(this.ringPos + i) % size] * this.window[i];
    }
    im.fill(0);

    // Fast Fourier transform
    for (var step = 1; step < size; step *= 2) {
        var skip = half / step;
        for (var i = 0; i < size; i += step * 2) {
            for (var j = 0; j < step; j++) {
                var wr = this.cos[j * skip], wi = -this.sin[j * skip];
                var a  = i + j, b = a + step;
                var tr = re[b] * wr - im[b] * wi;
                var ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }

    // Average power spectrum over time
    var total = 0;
    for (var i = 0; i < half; i++) {
        var p = re[i] * re[i] + im[i] * im[i];
        avg[i] += (p - avg[i]) * this.averaging;
        total  += avg[i];
    }

    // Find sharp peaks in the averaged spectrum
    var peaks = [];
    var nb    = this.neighbors;
    var floor = [];
    // Skip the search during silence
    var last  = total > half * size? half - 6 : 0;
    for (var i = Math.max(this.minBin, 5); i < last; i++) {
        var p = avg[i];
        // Must be the highest point within +-3 bins...
        if (p < avg[i-1] || p < avg[i-2] || p < avg[i-3]) continue;
        if (p <= avg[i+1] || p <= avg[i+2] || p <= avg[i+3]) continue;
        // ...be narrow, like a pure tone is...
        if (p < 10 * avg[i-5] || p < 10 * avg[i+5]) continue;
        // ...and stick out above what is typical around it
        floor.length = 0;
        for (var j = Math.max(1, i - nb); j <= Math.min(half - 1, i + nb); j++) {
            if (Math.abs(j - i) > 3) floor.push(avg[j]);
        }
        floor.sort(function(a, b) { return a - b; });
        if (p < this.threshold * floor[floor.length >> 1]) continue;
        // Refine frequency by fitting a parabola through the peak
        var l = Math.log(avg[i-1] + 1e-9), m = Math.log(p), r = Math.log(avg[i+1] + 1e-9);
        var d = l - 2 * m + r;
        peaks.push({ bin: i + (d < 0? 0.5 * (l - r) / d : 0), power: p });
    }

    // Strongest peaks get matched first
    peaks.sort(function(a, b) { return b.power - a.power; });

    // Match peaks to the tones we already know about
    var tracks = this.tracks;
    for (var t = 0; t < tracks.length; t++) tracks[t].seen = false;
    for (var n = 0; n < peaks.length && n < 2 * this.maxNotches; n++) {
        var peak  = peaks[n];
        var found = null;
        for (var t = 0; t < tracks.length; t++) {
            if (!tracks[t].seen && Math.abs(tracks[t].bin - peak.bin) <= 1) {
                // Until notched, a tone is not allowed to wander
                if (!tracks[t].active && Math.abs(tracks[t].origin - peak.bin) > this.steadiness) continue;
                found = tracks[t];
                break;
            }
        }
        if (found) {
            // Known tone, follow its frequency
            found.bin  = peak.bin;
            found.freq = peak.bin * this.binHz;
            found.seen = true;
        } else {
            // New tone, start watching it
            tracks.push({
                bin: peak.bin, freq: peak.bin * this.binHz, origin: peak.bin,
                hits: 0, misses: 0, seen: true, active: false, state: []
            });
        }
    }

    // Decide which tones to start or stop notching
    var active = tracks.filter(function(t) { return t.active; }).length;
    for (var t = tracks.length - 1; t >= 0; t--) {
        var track = tracks[t];
        if (track.seen) {
            track.hits++;
            track.misses = 0;
            // Tone has been there long enough, start notching it
            if (!track.active && track.hits >= this.engageHops && active < this.maxNotches) {
                track.active = true;
                active++;
                this.tuneNotch(track);
            }
            // Tone has drifted, retune its notch
            if (track.active && Math.abs(track.freq - track.tuned) > 0.5) {
                this.tuneNotch(track);
            }
        } else {
            track.misses++;
            // Tones may fade away for a while and come back: the longer
            // a tone has been there, the longer we wait for it.
            if (track.misses > Math.min(this.releaseHops, track.hits / 4)) {
                if (track.active) active--;
                tracks.splice(t, 1);
            }
        }
    }
};

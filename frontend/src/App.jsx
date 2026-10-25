import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'https://backend.voxlab.agamotto.tech').replace(/\/$/, '')

const getAudioUrl = (url) => {
  if (!url) return ''
  if (url.startsWith('http://') || url.startsWith('https://')) return url
  return `${API_BASE}${url}`
}

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('upload')

  // Voice Profile State
  const [activeVoice, setActiveVoice] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [activeFeature, setActiveFeature] = useState('waveform')

  // Voice Cloning State
  const [clonePrompt, setClonePrompt] = useState('Welcome everyone. This sentence is generated using our zero-shot voice cloning pipeline.')
  const [consentConfirmed, setConsentConfirmed] = useState(false)
  const [isCloning, setIsCloning] = useState(false)
  const [latestCloned, setLatestCloned] = useState(null)

  // AI Deepfake Detection State
  const [detectorSource, setDetectorSource] = useState('upload') // 'upload' | 'active' | 'cloned'
  const [detectorFile, setDetectorFile] = useState(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionReport, setDetectionReport] = useState(null)

  // Forensic Comparison State
  const [compareData, setCompareData] = useState(null)
  const [isComparing, setIsComparing] = useState(false)

  const fileInputRef = useRef(null)
  const detectorFileInputRef = useRef(null)

  // Load Pre-Bundled Reference Sample
  const handleLoadSample = async () => {
    setIsUploading(true)
    try {
      const res = await fetch(`${API_BASE}/api/audio/sample`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setActiveVoice(data)
    } catch (err) {
      alert('Error loading bundled sample: ' + err.message)
    } finally {
      setIsUploading(false)
    }
  }

  // File Upload Handler
  const handleFileUpload = async (file) => {
    if (!file) return
    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/api/audio/upload`, {
        method: 'POST',
        body: formData
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setActiveVoice(data)
    } catch (err) {
      alert('Upload error: ' + err.message)
    } finally {
      setIsUploading(false)
    }
  }

  // Voice Cloning Synthesis Handler
  const handleGenerateClone = async () => {
    if (!activeVoice || !clonePrompt.trim() || !consentConfirmed) return
    setIsCloning(true)
    try {
      const res = await fetch(`${API_BASE}/api/clone/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_id: activeVoice.audio_id,
          text_prompt: clonePrompt.trim(),
          consent: consentConfirmed
        })
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Synthesis failed')
      }
      const data = await res.json()
      setLatestCloned(data)
    } catch (err) {
      alert('Voice Cloning Error: ' + err.message)
    } finally {
      setIsCloning(false)
    }
  }

  // Deepfake Detection Handler
  const handleRunDetection = async () => {
    setIsDetecting(true)
    const formData = new FormData()

    if (detectorSource === 'upload') {
      if (!detectorFile) {
        alert('Please choose an audio file to inspect.')
        setIsDetecting(false)
        return
      }
      formData.append('file', detectorFile)
    } else if (detectorSource === 'active') {
      if (!activeVoice) {
        alert('No reference voice loaded.')
        setIsDetecting(false)
        return
      }
      formData.append('audio_id', activeVoice.audio_id)
    } else if (detectorSource === 'cloned') {
      if (!latestCloned) {
        alert('No cloned voice generated yet.')
        setIsDetecting(false)
        return
      }
      formData.append('audio_id', latestCloned.filename)
    }

    try {
      const res = await fetch(`${API_BASE}/api/security/detect`, {
        method: 'POST',
        body: formData
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Detection failed')
      }
      const report = await res.json()
      setDetectionReport(report)
    } catch (err) {
      alert('Detection Error: ' + err.message)
    } finally {
      setIsDetecting(false)
    }
  }

  // Forensics Side-by-Side Comparison Handler
  const handleLoadComparison = async () => {
    if (!activeVoice || !latestCloned) return
    setIsComparing(true)
    try {
      const res = await fetch(`${API_BASE}/api/security/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reference_id: activeVoice.audio_id,
          target_id: latestCloned.filename
        })
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setCompareData(data)
    } catch (err) {
      console.error('Comparison error:', err)
    } finally {
      setIsComparing(false)
    }
  }

  useEffect(() => {
    if (activeVoice && latestCloned) {
      handleLoadComparison()
    }
  }, [activeVoice, latestCloned])

  // Download Current Feature Plot
  const handleDownloadPlot = () => {
    if (!activeVoice || !activeVoice.plots[activeFeature]) return
    const link = document.createElement('a')
    link.download = `voxlab_${activeFeature}_plot.png`
    link.href = `data:image/png;base64,${activeVoice.plots[activeFeature]}`
    link.click()
  }

  return (
    <>
      {/* Header */}
      <header className="app-header">
        <div className="header-container">
          <div className="brand-group">
            <div className="brand-logo">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <path d="M12 2v20M17 5v14M7 8v8M2 11v2M22 11v2"></path>
              </svg>
            </div>
            <div>
              <h1 className="brand-title">VoxLab</h1>
              <p className="brand-subtitle">Voice Cloning &amp; Authenticity Detection</p>
            </div>
          </div>

          <div className="header-actions">
            <div className={`voice-status-badge ${activeVoice ? 'active' : 'inactive'}`}>
              <span className="status-dot"></span>
              <span>{activeVoice ? `Active: ${activeVoice.original_filename}` : 'No Voice Loaded'}</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs-wrapper">
          <div className="nav-tabs">
            <button className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
              <span className="tab-step">1</span>
              <span>Voice Onboarding</span>
            </button>
            <button className={`nav-tab ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>
              <span className="tab-step">2</span>
              <span>Audio Analysis</span>
            </button>
            <button className={`nav-tab ${activeTab === 'clone' ? 'active' : ''}`} onClick={() => setActiveTab('clone')}>
              <span className="tab-step">3</span>
              <span>Voice Cloning</span>
            </button>
            <button className={`nav-tab ${activeTab === 'detector' ? 'active' : ''}`} onClick={() => setActiveTab('detector')}>
              <span className="tab-step">4</span>
              <span>Deepfake Detector</span>
            </button>
            <button className={`nav-tab ${activeTab === 'compare' ? 'active' : ''}`} onClick={() => setActiveTab('compare')}>
              <span className="tab-step">5</span>
              <span>Voice Comparison</span>
            </button>
          </div>
        </nav>
      </header>

      {/* Main Content Area */}
      <main className="app-main">

        {/* TAB 1: Voice Onboarding */}
        {activeTab === 'upload' && (
          <section>
            <div className="section-header">
              <h2>Voice Onboarding</h2>
              <p>Upload a clean voice recording or try the pre-bundled sample to extract features and generate a voice profile.</p>
            </div>

            <div className="grid-2-col">
              {/* Upload Dropzone */}
              <div className="card">
                <div className="card-header">
                  <h3>Upload Voice Sample</h3>
                  <span className="badge">WAV, MP3, OGG, FLAC</span>
                </div>

                <div className="upload-dropzone" onClick={() => fileInputRef.current?.click()}>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept=".wav,.mp3,.ogg,.flac,.m4a"
                    hidden
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                  />
                  <div className="dropzone-icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                  </div>
                  <p className="dropzone-text"><strong>Click to browse</strong> or drag &amp; drop audio file</p>
                  <p className="dropzone-subtext">Optimal: 16 kHz clean speech with minimal reverberation</p>
                </div>

                <div className="divider-text"><span>OR</span></div>

                <div className="sample-action-box">
                  <button className="btn btn-secondary btn-full" onClick={handleLoadSample} disabled={isUploading}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"></circle>
                      <polygon points="10 8 16 12 10 16 10 8"></polygon>
                    </svg>
                    <span>{isUploading ? 'Processing Reference Audio...' : 'Load Pre-Bundled Reference Sample (One-Click)'}</span>
                  </button>
                  <p className="helper-text">Instantly loads the project's bundled reference speech for evaluation.</p>
                </div>
              </div>

              {/* Profile Card */}
              <div className="card">
                <div className="card-header">
                  <h3>Active Voice Profile</h3>
                  <span className={`badge ${activeVoice ? 'badge-success' : 'badge-muted'}`}>
                    {activeVoice ? 'Profile Ready' : 'Awaiting Audio'}
                  </span>
                </div>

                {!activeVoice ? (
                  <div className="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="12" y1="8" x2="12" y2="12"></line>
                      <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <p>No reference audio selected yet. Upload an audio recording or load the demo sample to continue.</p>
                  </div>
                ) : (
                  <div>
                    <div className="audio-player-card">
                      <div className="audio-meta">
                        <strong>{activeVoice.original_filename}</strong>
                        <span className="mono-tag">{activeVoice.features.duration_seconds}s</span>
                      </div>
                      <audio controls src={getAudioUrl(activeVoice.audio_url)} className="custom-audio-player"></audio>
                    </div>

                    <div className="quick-metrics-grid">
                      <div className="metric-card">
                        <span className="metric-label">Fundamental Pitch (F0)</span>
                        <span className="metric-value">{activeVoice.features.pitch.mean_f0_hz || 'N/A'} Hz</span>
                        <span className="metric-note">Range: {activeVoice.features.pitch.min_f0_hz} - {activeVoice.features.pitch.max_f0_hz} Hz</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">Spectral Brightness</span>
                        <span className="metric-value">{activeVoice.features.timbre.spectral_centroid_hz} Hz</span>
                        <span className="metric-note">Spectral Centroid</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">Harmonic Ratio</span>
                        <span className="metric-value">{activeVoice.features.timbre.harmonic_ratio}x</span>
                        <span className="metric-note">HPSS Index</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-label">Voiced Segments</span>
                        <span className="metric-value">{activeVoice.features.pitch.voiced_percentage}%</span>
                        <span className="metric-note">Speech Activity</span>
                      </div>
                    </div>

                    <button className="btn btn-primary btn-full" onClick={() => setActiveTab('analysis')}>
                      <span>View Audio Analysis &rarr;</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* TAB 2: Audio Analysis */}
        {activeTab === 'analysis' && (
          <section>
            <div className="section-header">
              <h2>Voice Feature Analysis</h2>
              <p>Explore extracted spectral, timbral, and pitch features of the reference voice.</p>
            </div>

            {/* Feature Selector Bar */}
            <div className="feature-selector-bar">
              {[
                { id: 'waveform', label: '1. Waveform' },
                { id: 'mel_spectrogram', label: '2. Mel-Spectrogram' },
                { id: 'mfcc', label: '3. MFCC Coefficients' },
                { id: 'pitch_f0', label: '4. Pitch (F0)' },
                { id: 'spectral_features', label: '5. Spectral Centroid & Rolloff' }
              ].map(item => (
                <button
                  key={item.id}
                  className={`feature-pill ${activeFeature === item.id ? 'active' : ''}`}
                  onClick={() => setActiveFeature(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Feature Plot Card */}
            {activeVoice ? (
              <div className="card">
                <div className="plot-header">
                  <div>
                    <h3>{activeVoice.explanations[activeFeature]?.name || activeFeature}</h3>
                    <p className="plot-subtitle">{activeVoice.explanations[activeFeature]?.description}</p>
                  </div>
                  <button className="btn btn-sm btn-secondary" onClick={handleDownloadPlot}>
                    Download Plot PNG
                  </button>
                </div>

                <div className="plot-image-container">
                  <img
                    src={`data:image/png;base64,${activeVoice.plots[activeFeature]}`}
                    alt="Audio Plot"
                    className="plot-img"
                  />
                </div>

                <div className="dsp-description-box">
                  <div className="desc-badge">Feature Explanation</div>
                  <p className="desc-main">{activeVoice.explanations[activeFeature]?.description}</p>
                  <div className="desc-significance">
                    <strong>Details: </strong>
                    <span>{activeVoice.explanations[activeFeature]?.dsp_significance}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="card empty-state">
                <p>No audio loaded yet. Please upload a voice sample in Tab 1 first.</p>
              </div>
            )}

            {/* Telemetry Breakdown Table */}
            {activeVoice && (
              <div className="card mt-24">
                <div className="card-header">
                  <h3>Voice Measurements Breakdown</h3>
                  <span className="badge">16 kHz Sample Rate</span>
                </div>
                <div className="telemetry-table-wrapper">
                  <table className="telemetry-table">
                    <thead>
                      <tr>
                        <th>Acoustic Metric</th>
                        <th>Measured Value</th>
                        <th>Standard Human Speech Range</th>
                        <th>Acoustic Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Fundamental Pitch (Mean F0)</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.pitch.mean_f0_hz} Hz</span></td>
                        <td>85–180 Hz (Male) | 165–260 Hz (Female)</td>
                        <td>Base rate of vocal fold vibration. Defines fundamental perceived pitch.</td>
                      </tr>
                      <tr>
                        <td><strong>Voiced Speech Ratio</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.pitch.voiced_percentage}%</span></td>
                        <td>60% – 85% in normal conversational speech</td>
                        <td>Fraction of signal where periodic vocal cord oscillations occur.</td>
                      </tr>
                      <tr>
                        <td><strong>Spectral Centroid</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.timbre.spectral_centroid_hz} Hz</span></td>
                        <td>1,200 – 2,800 Hz</td>
                        <td>Center of mass of spectrum. Quantifies vocal brightness and sharpness.</td>
                      </tr>
                      <tr>
                        <td><strong>Spectral Rolloff (85%)</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.timbre.spectral_rolloff_hz} Hz</span></td>
                        <td>2,500 – 5,500 Hz</td>
                        <td>Frequency below which 85% of total signal energy is contained.</td>
                      </tr>
                      <tr>
                        <td><strong>Harmonic Ratio (HPSS)</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.timbre.harmonic_ratio}x</span></td>
                        <td>&gt; 1.0 indicates strong harmonic content</td>
                        <td>Periodic vocal tract resonances versus unvoiced transient noise.</td>
                      </tr>
                      <tr>
                        <td><strong>Zero Crossing Rate (ZCR)</strong></td>
                        <td><span className="mono-tag" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{activeVoice.features.timbre.zero_crossing_rate}</span></td>
                        <td>0.03 – 0.12</td>
                        <td>Rate of waveform sign changes. Elevated during unvoiced fricatives.</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}

        {/* TAB 3: Voice Cloning */}
        {activeTab === 'clone' && (
          <section>
            <div className="section-header">
              <h2>Voice Cloning Studio</h2>
              <p>Synthesize custom text in the timbre, pitch, and style of your loaded voice profile.</p>
            </div>

            <div className="grid-2-col">
              <div className="card">
                <div className="card-header">
                  <h3>Speech Generation Prompt</h3>
                  <span className="badge badge-primary">Voice Cloner</span>
                </div>

                <div className="form-group">
                  <label className="form-label">Enter text to synthesize:</label>
                  <textarea
                    className="form-textarea"
                    rows="5"
                    value={clonePrompt}
                    onChange={(e) => setClonePrompt(e.target.value)}
                    placeholder="Type any sentence to synthesize in your cloned voice..."
                  />
                </div>

                <div className="prompt-chips-group">
                  <span className="chips-label">Quick Prompts:</span>
                  {[
                    "Hello, this sentence is generated in my cloned voice.",
                    "Testing voice synthesis and acoustic feature matching.",
                    "Voice models replicate pitch and timbre patterns."
                  ].map((p, idx) => (
                    <button key={idx} className="chip" onClick={() => setClonePrompt(p)}>
                      {idx === 0 ? 'Sample 1' : idx === 1 ? 'Sample 2' : 'Sample 3'}
                    </button>
                  ))}
                </div>

                <div className="consent-guard-box">
                  <label className="checkbox-container">
                    <input
                      type="checkbox"
                      checked={consentConfirmed}
                      onChange={(e) => setConsentConfirmed(e.target.checked)}
                    />
                    <span className="checkbox-text">
                      <strong>Authorization Confirmation:</strong> I confirm that I own this voice or possess permission to synthesize speech with this voice model.
                    </span>
                  </label>
                </div>

                <button
                  className="btn btn-primary btn-full mt-16"
                  onClick={handleGenerateClone}
                  disabled={!activeVoice || !clonePrompt.trim() || !consentConfirmed || isCloning}
                >
                  <span>{isCloning ? 'Synthesizing...' : 'Synthesize in My Voice'}</span>
                </button>
              </div>

              {/* Output Player Card */}
              <div className="card">
                <div className="card-header">
                  <h3>Synthesized Speech Output</h3>
                  <span className={`badge ${latestCloned ? 'badge-success' : 'badge-muted'}`}>
                    {latestCloned ? 'Synthesis Complete' : 'Idle'}
                  </span>
                </div>

                {!latestCloned ? (
                  <div className="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                      <line x1="12" y1="19" x2="12" y2="23"></line>
                      <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                    <p>Enter text and click Synthesize to generate speech conditioned on your reference voice profile.</p>
                  </div>
                ) : (
                  <div>
                    <div className="audio-player-card">
                      <div className="audio-meta">
                        <strong>{latestCloned.filename}</strong>
                        <span className="mono-tag">{latestCloned.duration_seconds}s</span>
                      </div>
                      <audio controls src={getAudioUrl(latestCloned.audio_url)} className="custom-audio-player"></audio>
                    </div>

                    <div className="similarity-box">
                      <div className="similarity-header">
                        <span>Voice Similarity Match</span>
                        <strong className="similarity-score">{latestCloned.voice_similarity_score}%</strong>
                      </div>
                      <div className="progress-bar-track">
                        <div
                          className="progress-bar-fill bg-success"
                          style={{ width: `${latestCloned.voice_similarity_score}%` }}
                        ></div>
                      </div>
                      <p className="similarity-note">Calculated based on acoustic feature similarity against reference audio.</p>
                    </div>

                    <div className="result-actions-row">
                      <a href={getAudioUrl(latestCloned.audio_url)} download={latestCloned.filename} className="btn btn-secondary flex-1">
                        Download Cloned WAV
                      </a>
                      <button
                        className="btn btn-outline"
                        onClick={() => {
                          setActiveTab('detector')
                          setDetectorSource('cloned')
                        }}
                      >
                        Test in Detector &rarr;
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* TAB 4: Deepfake Detector */}
        {activeTab === 'detector' && (
          <section>
            <div className="section-header">
              <h2>Voice Authenticity Detector</h2>
              <p>Analyze audio recordings to determine whether a voice is human speech or AI-generated.</p>
            </div>

            <div className="grid-2-col">
              <div className="card">
                <div className="card-header">
                  <h3>Select Audio to Verify</h3>
                  <span className="badge">Audio Analysis</span>
                </div>

                <div className="source-toggle-group">
                  <button
                    className={`source-btn ${detectorSource === 'upload' ? 'active' : ''}`}
                    onClick={() => setDetectorSource('upload')}
                  >
                    Upload New File
                  </button>
                  <button
                    className={`source-btn ${detectorSource === 'active' ? 'active' : ''}`}
                    onClick={() => {
                      if (!activeVoice) alert('Please upload a reference voice in Tab 1 first.')
                      else setDetectorSource('active')
                    }}
                  >
                    Active Voice
                  </button>
                  <button
                    className={`source-btn ${detectorSource === 'cloned' ? 'active' : ''}`}
                    onClick={() => {
                      if (!latestCloned) alert('Please synthesize a cloned sample in Tab 3 first.')
                      else setDetectorSource('cloned')
                    }}
                  >
                    Latest Cloned Speech
                  </button>
                </div>

                {detectorSource === 'upload' && (
                  <div className="upload-dropzone mt-16" onClick={() => detectorFileInputRef.current?.click()}>
                    <input
                      type="file"
                      ref={detectorFileInputRef}
                      accept=".wav,.mp3,.ogg,.flac,.m4a"
                      hidden
                      onChange={(e) => setDetectorFile(e.target.files?.[0] || null)}
                    />
                    <div className="dropzone-icon">
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                      </svg>
                    </div>
                    <p className="dropzone-text"><strong>Choose audio</strong> to analyze for AI deepfake traces</p>
                    <p className="dropzone-subtext">
                      {detectorFile ? `Selected: ${detectorFile.name}` : 'Supports WAV, MP3, OGG, FLAC'}
                    </p>
                  </div>
                )}

                <button
                  className="btn btn-primary btn-full mt-16"
                  onClick={handleRunDetection}
                  disabled={isDetecting}
                >
                  <span>{isDetecting ? 'Analyzing Audio...' : 'Run Authenticity Analysis'}</span>
                </button>
              </div>

              {/* Detection Results */}
              <div className="card">
                <div className="card-header">
                  <h3>Authenticity Analysis Verdict</h3>
                  <span className={`badge ${detectionReport ? (detectionReport.is_ai_generated ? 'badge-danger' : 'badge-success') : 'badge-muted'}`}>
                    {detectionReport ? (detectionReport.is_ai_generated ? 'AI-Generated Voice' : 'Authentic Human Voice') : 'Awaiting Scan'}
                  </span>
                </div>

                {!detectionReport ? (
                  <div className="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    </svg>
                    <p>Select an audio source and click Run Analysis to view the deepfake probability and forensic breakdown.</p>
                  </div>
                ) : (
                  <div>
                    <div className={`verdict-hero-card ${detectionReport.is_ai_generated ? 'verdict-ai' : 'verdict-human'}`}>
                      <div className="verdict-badge-large">{detectionReport.verdict}</div>
                      <div className="authenticity-gauge-row">
                        <span>Human Authenticity Score:</span>
                        <strong className="gauge-value">{detectionReport.authenticity_score}%</strong>
                      </div>
                      <p className="verdict-caption">
                        {detectionReport.is_ai_generated
                          ? 'Telltale neural vocoder cutoff, acoustic phase artifacts, or high CNN generative energy detected.'
                          : 'Natural vocal micro-jitter, authentic glottal excitation, and healthy spectral roll-off verified.'}
                      </p>
                    </div>

                    <div className="probability-split-box">
                      <div className="prob-row">
                        <span>Authentic Human Voice</span>
                        <span className="prob-val">{(detectionReport.probabilities.human * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar-track">
                        <div
                          className="progress-bar-fill bg-success"
                          style={{ width: `${(detectionReport.probabilities.human * 100).toFixed(1)}%` }}
                        ></div>
                      </div>

                      <div className="prob-row mt-12">
                        <span>AI-Generated / Deepfake</span>
                        <span className="prob-val">{(detectionReport.probabilities.ai_generated * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar-track">
                        <div
                          className="progress-bar-fill bg-danger"
                          style={{ width: `${(detectionReport.probabilities.ai_generated * 100).toFixed(1)}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="forensic-checklist">
                      {detectionReport.indicators.map((ind, i) => (
                        <div key={i} className="checklist-item">
                          <div className={`checklist-status-badge status-${ind.status}`}>{ind.status.toUpperCase()}</div>
                          <div className="checklist-content">
                            <div className="checklist-title">
                              <span>{ind.name}</span>
                              <span className="checklist-score">{ind.score}</span>
                            </div>
                            <p className="checklist-desc">{ind.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* TAB 5: Voice Comparison */}
        {activeTab === 'compare' && (
          <section>
            <div className="section-header">
              <h2>Side-by-Side Voice Comparison</h2>
              <p>Compare reference voice against generated speech across pitch, spectral, and timbral features.</p>
            </div>

            <div className="card">
              <div className="card-header">
                <h3>Voice Comparison Benchmark</h3>
                <button className="btn btn-sm btn-secondary" onClick={handleLoadComparison} disabled={isComparing}>
                  {isComparing ? 'Computing...' : 'Re-calculate Distance'}
                </button>
              </div>

              {!compareData ? (
                <div className="empty-state">
                  <p>Please upload a reference voice and synthesize at least one cloned sample to unlock side-by-side comparison.</p>
                </div>
              ) : (
                <div>
                  <div className="grid-2-col">
                    <div className="compare-column">
                      <div className="column-title">
                        <span className="badge badge-success">Reference Voice</span>
                        <strong>{activeVoice?.original_filename}</strong>
                      </div>
                      <div className="compare-metrics-list">
                        <div className="comp-metric-row"><span>Mean Pitch (F0):</span><span className="comp-val">{compareData.reference_profile.mean_f0_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Pitch Range:</span><span className="comp-val">{compareData.reference_profile.min_f0_hz} - {compareData.reference_profile.max_f0_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Spectral Brightness:</span><span className="comp-val">{compareData.reference_profile.spectral_centroid_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Vocal Register:</span><span className="comp-val">{compareData.reference_profile.gender_estimate.toUpperCase()}</span></div>
                      </div>
                    </div>

                    <div className="compare-column">
                      <div className="column-title">
                        <span className="badge badge-primary">Cloned Speech</span>
                        <strong>{latestCloned?.filename}</strong>
                      </div>
                      <div className="compare-metrics-list">
                        <div className="comp-metric-row"><span>Mean Pitch (F0):</span><span className="comp-val">{compareData.target_profile.mean_f0_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Pitch Range:</span><span className="comp-val">{compareData.target_profile.min_f0_hz} - {compareData.target_profile.max_f0_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Spectral Brightness:</span><span className="comp-val">{compareData.target_profile.spectral_centroid_hz} Hz</span></div>
                        <div className="comp-metric-row"><span>Conditioned Register:</span><span className="comp-val">{compareData.target_profile.gender_estimate.toUpperCase()}</span></div>
                      </div>
                    </div>
                  </div>

                  <div className="compare-summary-banner mt-24">
                    <div className="summary-metric">
                      <span className="summary-label">Acoustic Similarity Match</span>
                      <strong className="summary-value">{compareData.similarity_score_percentage}%</strong>
                    </div>
                    <div className="summary-metric">
                      <span className="summary-label">Pitch Delta (ΔF0)</span>
                      <strong className="summary-value">{compareData.pitch_difference_hz} Hz</strong>
                    </div>
                    <div className="summary-metric">
                      <span className="summary-label">Spectral Timbre Delta</span>
                      <strong className="summary-value">{compareData.spectral_centroid_diff_hz} Hz</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-container">
          <p>VoxLab &bull; Speech Processing &amp; Voice Synthesis</p>
        </div>
      </footer>
    </>
  )
}

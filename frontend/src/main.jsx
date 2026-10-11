import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const root = document.getElementById('root')

if (!root) throw new Error('The React root element was not found.')

const renderApp = (component) => createRoot(root).render(component)

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

function formatLabel(label) {
  return label.replaceAll('_', ' ')
}

function formatPercent(score) {
  return `${Math.round(score * 100)}%`
}

function getDecision(result) {
  const action = result?.routing?.action || result?.status || ''
  if (action.includes('reject')) return 'REJECTED'
  if (action.includes('approve')) return 'APPROVED'
  if (action.includes('queue') || action.includes('review') || action.includes('uncertain')) {
    return 'NEEDS REVIEW'
  }
  return action.replaceAll('_', ' ').toUpperCase() || 'UNKNOWN'
}

function getDecisionTone(result) {
  const decision = getDecision(result)
  if (decision === 'REJECTED') return 'danger'
  if (decision === 'APPROVED') return 'success'
  return 'warning'
}

function getPrediction(result) {
  if (!result) return null
  return result.prediction || result.text_prediction || result.image_prediction
}

function getImageDecision(result) {
  // If the image prediction has a dedicated decision field, use it
  const prediction = getPrediction(result)
  if (prediction?.decision) {
    return prediction.decision
  }
  // Fallback to the routing action
  return getDecision(result)
}

function getImageDecisionTone(result) {
  const decision = getImageDecision(result)
  if (decision === 'NSFW' || decision === 'REJECTED') return 'danger'
  if (decision === 'SAFE' || decision === 'APPROVED') return 'success'
  return 'warning'
}

function ResultPanel({ result, error }) {
  const prediction = getPrediction(result)
  const labels = prediction?.labels || []
  const decision = getImageDecision(result)
  const tone = getImageDecisionTone(result)
  const isImageResult = prediction?.decision !== undefined

  return (
    <section className="panel results-panel" aria-live="polite">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Model output</p>
          <h2>Results</h2>
        </div>
        {result && <span className={`decision-pill ${tone}`}>{decision}</span>}
      </div>

      {!result && !error && (
        <div className="empty-state">
          <span className="empty-icon">✦</span>
          <p>Run a text or image check to see what your model predicts.</p>
        </div>
      )}

      {error && (
        <div className="error-box" role="alert">
          <strong>Request failed</strong>
          <span>{error}</span>
        </div>
      )}

      {result && prediction && (
        <div className="result-content">
          <div className="result-summary">
            <div>
              <span className="summary-label">Overall decision</span>
              <strong className={`summary-value ${tone}`}>{decision}</strong>
            </div>
            <div className="top-score">
              <span className="summary-label">Top signal</span>
              <strong>{formatLabel(prediction.top_label)}</strong>
              <span>{formatPercent(prediction.top_score)} confidence</span>
            </div>
          </div>

          {/* Image-specific NSFW score display */}
          {isImageResult && (
            <div className="result-summary" style={{ marginTop: '0.75rem' }}>
              <div>
                <span className="summary-label">NSFW Score</span>
                <strong className="summary-value">{formatPercent(prediction.nsfw_score || 0)}</strong>
              </div>
              {prediction.method && (
                <div>
                  <span className="summary-label">Method</span>
                  <strong>{formatLabel(prediction.method)}</strong>
                </div>
              )}
            </div>
          )}

          <div className="score-list">
            {labels.map((item) => (
              <div className="score-row" key={item.label}>
                <div className="score-name">
                  <span className={item.predicted ? 'flag-dot active' : 'flag-dot'} />
                  <span>{formatLabel(item.label)}</span>
                  {item.predicted && <span className="flagged-label">flagged</span>}
                </div>
                <div className="score-track" aria-hidden="true">
                  <span style={{ width: `${item.score * 100}%` }} />
                </div>
                <strong>{formatPercent(item.score)}</strong>
              </div>
            ))}
          </div>

          <div className="result-meta">
            <span>Action: {formatLabel(result.routing?.action || result.status)}</span>
            {prediction.model_version && <span>Model: {prediction.model_version}</span>}
          </div>
        </div>
      )}
    </section>
  )
}

function App() {
  const [text, setText] = useState('')
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState('')
  const fileInputRef = useRef(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  function handleImageChange(event) {
    const file = event.target.files?.[0]
    if (!file) return

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  async function requestModeration(path, options) {
    const response = await fetch(`${API_BASE_URL}${path}`, options)
    const body = await response.json().catch(() => ({}))
    if (!response.ok) {
      const detail = Array.isArray(body?.detail)
        ? body.detail.map((item) => item.msg).join(', ')
        : body?.detail
      throw new Error(detail || `The backend returned HTTP ${response.status}.`)
    }
    return body
  }

  async function moderateText(event) {
    event.preventDefault()
    if (!text.trim()) {
      setError('Enter some text before moderating it.')
      return
    }

    setLoading('text')
    setError('')
    setResult(null)
    try {
      const body = await requestModeration('/moderate/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      setResult(body)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading('')
    }
  }

  async function moderateImage() {
    if (!selectedImage) {
      setError('Choose an image before moderating it.')
      return
    }

    setLoading('image')
    setError('')
    setResult(null)
    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const body = await requestModeration('/moderate/image', {
        method: 'POST',
        body: formData,
      })
      setResult(body)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading('')
    }
  }

  return (
    <main className="app-shell">
      <div className="page-wrap mx-auto">
        <header className="hero">
          <div className="brand-mark">CM</div>
          <div>
            <p className="eyebrow">Local model playground</p>
            <h1>Content Moderation Tester</h1>
            <p className="hero-copy">Send text or an image to your existing moderation API and inspect the prediction.</p>
          </div>
        </header>

        <div className="workspace">
          <div className="inputs-column">
            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Text model</p>
                  <h2>Text Moderation</h2>
                </div>
                <span className="section-number">01</span>
              </div>
              <form onSubmit={moderateText}>
                <label className="field-label" htmlFor="moderation-text">Content to check</label>
                <textarea
                  id="moderation-text"
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  placeholder="Paste a message, comment, or post here..."
                  maxLength={10000}
                />
                <div className="field-footer">
                  <span>Up to 10,000 characters</span>
                  <span>{text.length}/10,000</span>
                </div>
                <button className="primary-button" aria-busy={loading === 'text'} disabled={loading !== ''} type="submit">
                  {loading === 'text' ? <><span className="spinner" role="status" aria-label="Loading" /> Checking text...</> : <>Moderate Text <span>→</span></>}
                </button>
              </form>
            </section>

            <section className="panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Vision model</p>
                  <h2>Image Moderation</h2>
                </div>
                <span className="section-number">02</span>
              </div>
              <input ref={fileInputRef} className="hidden-input" type="file" accept="image/*" onChange={handleImageChange} />
              <button className="upload-zone" type="button" onClick={() => fileInputRef.current?.click()}>
                {previewUrl ? (
                  <img src={previewUrl} alt="Selected image preview" />
                ) : (
                  <span className="upload-placeholder"><span className="upload-icon">↥</span><strong>Choose an image</strong><small>PNG, JPG, or WEBP</small></span>
                )}
              </button>
              {selectedImage && <p className="file-name">{selectedImage.name}</p>}
              <button className="secondary-button" aria-busy={loading === 'image'} disabled={loading !== ''} type="button" onClick={moderateImage}>
                {loading === 'image' ? <><span className="spinner dark" role="status" aria-label="Loading" /> Checking image...</> : <>Moderate Image <span>→</span></>}
              </button>
            </section>
          </div>

          <ResultPanel result={result} error={error} />
        </div>

        <footer className="footer-note">Connected to <code>{API_BASE_URL}</code></footer>
      </div>
    </main>
  )
}

export default App

renderApp(<App />)

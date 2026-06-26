export default function Memorial({ onExit }) {
  return (
    <div className="page">
      <header className="page-head">
        <button className="admin-btn" onClick={onExit} aria-label="Back to chat">
          ← Chat
        </button>
        <h1>In Memoriam</h1>
        <span />
      </header>

      <div className="page-body">
        <div className="mem-hero">
          <p className="mem-name">Dr. Raymond Peat</p>
          <p className="mem-years">1936 – 2022</p>
          <p className="mem-role">Pioneering bioenergetic researcher and independent scholar</p>
        </div>

        <blockquote className="mem-quote">
          “Energy and structure are interdependent at every level of organization.”
          <cite>— Ray Peat</cite>
        </blockquote>

        <h3>Remembering a visionary</h3>
        <p>
          For more than five decades, Dr. Ray Peat devoted himself to understanding how cells produce
          and use energy — and what that means for human health. An independent researcher with a
          Ph.D. in biology from the University of Oregon, he challenged conventional thinking with his
          bioenergetic view of physiology, bridging biochemistry and everyday, practical health.
        </p>
        <p>
          His interviews, articles, and newsletters continue to teach. PeatLearn exists to keep his
          words accessible and grounded — every answer here is drawn from his own writing.
        </p>

        <h3>The bioenergetic view</h3>
        <div className="mem-principles">
          <div className="mem-card">
            <h4>Energy as the central variable</h4>
            <p>A cell's ability to produce and use energy shapes its capacity to maintain structure, function, and resist stress.</p>
          </div>
          <div className="mem-card">
            <h4>Protective factors</h4>
            <p>Thyroid hormone, progesterone, adequate sugar, saturated fats, and key minerals support energy production.</p>
          </div>
          <div className="mem-card">
            <h4>The thyroid connection</h4>
            <p>Thyroid hormone tunes the cell's machinery for efficient energy production with minimal waste.</p>
          </div>
          <div className="mem-card">
            <h4>CO₂ — not just waste</h4>
            <p>Carbon dioxide improves oxygen delivery, stabilizes enzymes, and helps protect the cell.</p>
          </div>
        </div>

        <p className="page-note">
          PeatLearn is an unofficial, educational tribute — not affiliated with Ray Peat or his
          estate. Answers reflect his published views and are not medical advice.
        </p>
      </div>
    </div>
  );
}

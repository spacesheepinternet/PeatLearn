export default function Privacy({ onExit }) {
  return (
    <div className="page">
      <header className="page-head">
        <button className="admin-btn" onClick={onExit} aria-label="Back to chat">
          ← Chat
        </button>
        <h1>Privacy Policy</h1>
        <span />
      </header>

      <div className="page-body">
        <p className="page-updated">Last updated: June 25, 2026</p>

        <p>
          PeatLearn is a free, educational project that answers questions using Dr. Ray Peat's
          published work. This page explains, in plain language, what information we collect and
          why. It is not legal or medical advice.
        </p>

        <h2>What we collect</h2>
        <ul>
          <li>The <strong>questions you ask</strong> and the <strong>answers</strong> you receive.</li>
          <li>
            Basic technical details about each request: the time, how long it took, which source
            documents were cited, and a <strong>one-way hashed identifier derived from your IP
            address</strong>.
          </li>
        </ul>

        <h2>What we do not collect</h2>
        <ul>
          <li>
            <strong>We do not store your actual IP address.</strong> We only keep a scrambled
            (hashed) version that lets us tell repeat visits apart and spot abuse — it cannot be
            turned back into your real address.
          </li>
          <li>No account, name, email, or precise location.</li>
          <li>
            No advertising or third-party tracking cookies. Your browser's local storage is used
            only to remember your light/dark theme preference.
          </li>
          <li>We never sell or rent your data.</li>
        </ul>

        <h2>How your question is processed</h2>
        <p>
          To produce an answer, your question is sent to third-party AI services (such as Google's
          Gemini) and a search index, which process it to generate and fact-check the response.
          Because of this, please <strong>don't include personal, identifying, or sensitive
          details</strong> in your questions.
        </p>

        <h2>Why we keep a log</h2>
        <p>
          We keep a record of questions and answers to monitor answer quality and safety, catch
          mistakes, and improve the service. Logs are retained only as long as they're useful for
          that purpose.
        </p>

        <h2>Children</h2>
        <p>
          PeatLearn is not directed to children under 13, and we do not knowingly collect their
          information.
        </p>

        <h2>Your choices</h2>
        <p>
          You can ask us to delete records associated with your use of the site. To make a request
          or ask a privacy question, contact us at{" "}
          <strong>privacy@peatlearn.com</strong>.
        </p>

        <h2>Changes</h2>
        <p>
          We may update this policy as the project evolves. The date at the top reflects the latest
          version.
        </p>

        <p className="page-note">
          PeatLearn is an unofficial, educational project and is not affiliated with Ray Peat or his
          estate. Answers reflect his views and are not medical advice.
        </p>
      </div>
    </div>
  );
}

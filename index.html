export default function TruthScanPro() {
  
  // Backend API Ready
  // Connect this UI with FastAPI backend running on port 8000
  // Backend should expose POST /predict endpoint
  // News verification APIs:
  // - NewsAPI
  // - GNews
  // - Google Fact Check API

  const sampleSources = [
    {
      name: "Reuters",
      title: "Global leaders discuss climate policy at summit",
      credibility: "Trusted"
    },
    {
      name: "BBC News",
      title: "Researchers publish verified medical findings",
      credibility: "Trusted"
    },
    {
      name: "Associated Press",
      title: "Economic report confirms inflation slowdown",
      credibility: "Trusted"
    }
  ];

  async function analyzeNews() {
    try {
      const text = document.querySelector("textarea")?.value;

      if (!text || text.trim() === "") {
        alert("Please enter news text.");
        return;
      }

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: text
        })
      });

      const data = await response.json();

      console.log("Prediction Result:", data);

      if (data.verification && data.verification.length > 0) {
        const verifiedSources = data.verification
          .map(
            (article) =>
              `${article.source}: ${article.title}`
          )
          .join("

");

        alert(
          `Prediction: ${data.prediction}
Confidence: ${data.confidence}%

Verified Sources:

${verifiedSources}`
        );
      } else {
        alert(
          `Prediction: ${data.prediction}
Confidence: ${data.confidence}%

No trusted verification sources found.`
        );
      }
    } catch (err) {
      console.error(err);
      alert("Backend or verification API not running.");
    }
  }

  // FastAPI backend should:
  // 1. Run ML fake news detection
  // 2. Query NewsAPI / GNews
  // 3. Compare trusted articles
  // 4. Return Reuters/BBC/AP verification


  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,#312e81,transparent_35%),radial-gradient(circle_at_bottom_right,#065f46,transparent_35%)] opacity-40"></div>

      <nav className="relative z-10 flex items-center justify-between px-8 py-6 border-b border-white/10 backdrop-blur-xl">
        <div>
          <h1 className="text-3xl font-black tracking-tight">
            Truth<span className="text-violet-400">Scan</span>
          </h1>
          <p className="text-xs text-zinc-400 mt-1">AI Fake News Verification Platform</p>
        </div>

        <div className="flex gap-3 items-center">
          <div className="px-4 py-2 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-sm">
            ML + Live Verification
          </div>
        </div>
      </nav>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-14 grid lg:grid-cols-2 gap-10 items-start">
        <div>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-violet-500/20 bg-violet-500/10 text-violet-300 text-sm mb-6">
            Real-time AI Detection
          </div>

          <h2 className="text-6xl font-black leading-tight tracking-tight mb-6">
            Detect Fake News
            <span className="block text-violet-400">Professionally</span>
          </h2>

          <p className="text-zinc-400 text-lg leading-relaxed max-w-xl mb-8">
            Verify headlines using machine learning, trusted news sources,
            and AI-powered credibility analysis.
          </p>

          <div className="grid grid-cols-3 gap-4 mb-10">
            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-xl">
              <h3 className="text-3xl font-black text-emerald-400">96%</h3>
              <p className="text-zinc-400 text-sm mt-2">Model Accuracy</p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-xl">
              <h3 className="text-3xl font-black text-violet-400">3+</h3>
              <p className="text-zinc-400 text-sm mt-2">Trusted Sources</p>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-xl">
              <h3 className="text-3xl font-black text-cyan-400">AI</h3>
              <p className="text-zinc-400 text-sm mt-2">Verification Engine</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            {[
              "FastAPI",
              "Scikit-learn",
              "NLP",
              "NewsAPI",
              "TF-IDF",
              "React"
            ].map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-zinc-300"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-3xl backdrop-blur-2xl p-8 shadow-2xl shadow-violet-500/10">
          <div className="flex items-center gap-2 mb-6">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="ml-4 text-zinc-400 text-sm">truthscan-ai-engine</span>
          </div>

          <textarea
            placeholder="Paste a news headline or article here..."
            className="w-full h-44 bg-black/40 border border-white/10 rounded-2xl p-5 outline-none focus:border-violet-500 text-zinc-200 resize-none"
            defaultValue="Scientists confirm new renewable energy breakthrough reducing electricity costs globally"
          ></textarea>

          <button
            onClick={analyzeNews}
            className="w-full mt-5 py-4 rounded-2xl bg-violet-600 hover:bg-violet-500 transition-all font-bold text-lg shadow-lg shadow-violet-500/20"
          >
            Analyze News
          </button>

          <div className="mt-8 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-3xl font-black text-emerald-400">REAL</h3>
                <p className="text-zinc-400 mt-1">Verified by trusted sources</p>
              </div>

              <div className="px-4 py-2 rounded-full bg-emerald-500/20 text-emerald-300 text-sm font-semibold">
                VERIFIED
              </div>
            </div>

            <div className="mb-5">
              <div className="flex justify-between mb-2 text-sm text-zinc-400">
                <span>Confidence Score</span>
                <span className="text-white font-semibold">94%</span>
              </div>

              <div className="w-full bg-black/40 rounded-full h-3 overflow-hidden">
                <div className="h-full w-[94%] bg-emerald-400 rounded-full"></div>
              </div>
            </div>

            <div className="space-y-3">
              {sampleSources.map((source) => (
                <div
                  key={source.name}
                  className="bg-black/30 border border-white/10 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-white">{source.name}</h4>
                    <span className="text-xs px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300">
                      {source.credibility}
                    </span>
                  </div>

                  <p className="text-zinc-400 text-sm">{source.title}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      <section className="relative z-10 max-w-7xl mx-auto px-6 pb-20">
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl">
            <h3 className="text-2xl font-black mb-4 text-violet-400">AI Detection</h3>
            <p className="text-zinc-400 leading-relaxed">
              NLP and machine learning models analyze news content using TF-IDF and advanced classification techniques.
            </p>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl">
            <h3 className="text-2xl font-black mb-4 text-emerald-400">Live Verification</h3>
            <p className="text-zinc-400 leading-relaxed">
              News is cross-checked against Reuters, BBC, and Associated Press for real-world verification.
            </p>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-3xl p-8 backdrop-blur-xl">
            <h3 className="text-2xl font-black mb-4 text-cyan-400">Credibility Engine</h3>
            <p className="text-zinc-400 leading-relaxed">
              Source trust score, sensational language analysis, and confidence scoring improve reliability.
            </p>
          </div>
        </div>
      </section>

      <footer className="relative z-10 border-t border-white/10 py-8 text-center text-zinc-500 text-sm">
        TruthScan Pro — AI Powered Fake News Verification Platform
      </footer>
    </div>
  );
}

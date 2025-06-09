export default function AboutPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-3xl w-full bg-gray-900 rounded-xl shadow-lg p-8 border border-green-500 flex flex-col">
        <h1 className="text-4xl font-bold text-green-400 mb-4 text-center">About Drone Crop</h1>
        <p className="text-gray-200 text-lg mb-6 text-center">
          <span className="font-semibold text-green-400">Our Mission</span><br />
          Drone Crop aims to make advanced agricultural insights accessible to every farmer, regardless of their background or digital literacy. By transforming complex drone and satellite data into clear, actionable advice in local languages, we empower farmers to make informed decisions for healthier crops and better yields.
        </p>
        <p className="text-gray-200 text-lg mb-6 text-center">
          <span className="font-semibold text-green-400">Why We Exist</span><br />
          Many farmers face challenges understanding technical crop health data. Our platform bridges this gap by providing personalized, easy-to-understand crop health reports, supporting timely actions like nutrient management and pest control. We believe that technology should serve everyone, and our solutions are designed to be inclusive and practical for real-world farming needs.
        </p>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-green-400 mb-2 text-center">What We Offer</h2>
          <ul className="list-disc list-inside text-gray-100 space-y-2">
            <li>
              AI-powered summaries of drone and satellite imagery, highlighting key crop health issues.
            </li>
            <li>
              Support for multiple Indian languages, ensuring accessibility for farmers across regions.
            </li>
            <li>
              Seamless integration with agri-drone platforms and mobile advisory services.
            </li>
            <li>
              Curated datasets from Indian farms to improve and validate our crop analysis models.
            </li>
          </ul>
        </div>
        <div className="mt-8 text-center">
          <a
            href="/"
            className="inline-block px-6 py-2 rounded-lg bg-green-500 text-black font-semibold hover:bg-green-400 transition"
          >
            Back to Home
          </a>
        </div>
      </div>
    </div>
  );
}
export default function ServicesPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-3xl w-full bg-gray-900 rounded-xl shadow-lg p-8 border border-green-500 flex flex-col">
        
        <h1 className="text-4xl font-bold text-green-400 mb-4 text-center">Our Services</h1>
        <ul className="list-disc list-inside text-gray-100 space-y-6 text-lg">
          <li>
            <span className="font-semibold text-green-400">AI-Powered Crop Health Summaries:</span> 
            <br />
            Our generative AI model transforms drone and satellite imagery into natural-language crop health reports, highlighting issues like disease spread, nutrient deficiency, water stress, and weed coverage.
          </li>
          <li>
            <span className="font-semibold text-green-400">Local Language Support:</span>
            <br />
            We generate crop health insights in Hindi, Kannada, Tamil, Bengali, and other Indian languages, making our services accessible to farmers across diverse regions.
          </li>
          <li>
            <span className="font-semibold text-green-400">Seamless Platform Integration:</span>
            <br />
            Our integration modules connect with agri-drone platforms and mobile advisory systems, including Kisan Call Centers and agri-tech apps, for smooth delivery of actionable advice.
          </li>
          <li>
            <span className="font-semibold text-green-400">Curated Agricultural Datasets:</span>
            <br />
            We maintain a labeled dataset of annotated drone images from Indian farms, supporting the training and validation of region-specific crop health analysis models.
          </li>
        </ul>
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
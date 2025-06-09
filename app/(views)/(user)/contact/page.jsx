export default function ContactPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-xl w-full bg-gray-900 rounded-xl shadow-lg p-8 border border-green-500 flex flex-col">
        <div className="flex justify-center mb-6">
          {/* Contact SVG */}
          <svg width="44" height="44" viewBox="0 0 48 48" fill="none" className="mx-2" xmlns="http://www.w3.org/2000/svg">
            <circle cx="24" cy="24" r="22" stroke="#22c55e" strokeWidth="4" fill="#111827"/>
            <path d="M16 20a8 8 0 0 1 16 0c0 4.418-3.582 8-8 8s-8-3.582-8-8z" fill="#22c55e"/>
            <rect x="18" y="30" width="12" height="4" rx="2" fill="#22c55e"/>
          </svg>
        </div>
        <h1 className="text-4xl font-bold text-green-400 mb-4 text-center">Contact Us</h1>
        <p className="text-gray-200 text-lg mb-6 text-center">
          Have questions or want to collaborate? Reach out to the Drone Crop team using the form below.
        </p>
        <form className="flex flex-col gap-4">
          <input
            type="text"
            placeholder="Your Name"
            className="rounded-lg px-4 py-2 bg-black border border-green-500 text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-400"
            required
          />
          <input
            type="email"
            placeholder="Your Email"
            className="rounded-lg px-4 py-2 bg-black border border-green-500 text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-400"
            required
          />
          <textarea
            placeholder="Your Message"
            rows={5}
            className="rounded-lg px-4 py-2 bg-black border border-green-500 text-gray-100 focus:outline-none focus:ring-2 focus:ring-green-400"
            required
          />
          <button
            type="submit"
            className="mt-2 px-6 py-2 rounded-lg bg-green-500 text-black font-semibold hover:bg-green-400 transition"
          >
            Send Message
          </button>
        </form>
        <div className="mt-8 text-center text-gray-400 text-sm">
          Or email us at <a href="mailto:contact@dronecrop.in" className="text-green-400 underline">contact@dronecrop.in</a>
        </div>
      </div>
    </div>
  );
}
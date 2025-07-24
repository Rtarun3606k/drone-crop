// ./app/api-doc/page.jsx
import { getApiDocs } from "../lib/swagger";
import ReactSwagger from "./react-swagger";

export default async function ApiDocPage() {
  const spec = await getApiDocs();
  return (
    <section className="min-h-screen bg-zinc-900 text-white py-8 px-4 md:px-12">
      <div className="max-w-7xl mx-auto bg-white rounded-2xl shadow-lg p-6 dark:bg-zinc-800">
        <ReactSwagger spec={spec} />
      </div>
    </section>
  );
}

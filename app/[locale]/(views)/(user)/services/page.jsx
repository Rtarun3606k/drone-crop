"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";

export default function ServicesPage() {
  const t = useTranslations("services");

  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-3xl w-full bg-gray-900 rounded-xl shadow-lg p-8 border border-green-500 flex flex-col">
        <h1 className="text-4xl font-bold text-green-400 mb-4 text-center">
          {t("title")}
        </h1>
        <ul className="list-disc list-inside text-gray-100 space-y-6 text-lg">
          <li>
            <span className="font-semibold text-green-400">
              {t("ai_title")}:
            </span>
            <br />
            {t("ai_description")}
          </li>
          <li>
            <span className="font-semibold text-green-400">
              {t("language_title")}:
            </span>
            <br />
            {t("language_description")}
          </li>
          <li>
            <span className="font-semibold text-green-400">
              {t("integration_title")}:
            </span>
            <br />
            {t("integration_description")}
          </li>
          <li>
            <span className="font-semibold text-green-400">
              {t("data_title")}:
            </span>
            <br />
            {t("data_description")}
          </li>
        </ul>
        <div className="mt-8 text-center">
          <Link
            href="/"
            className="inline-block px-6 py-2 rounded-lg bg-green-500 text-black font-semibold hover:bg-green-400 transition"
          >
            {t("back_to_home")}
          </Link>
        </div>
      </div>
    </div>
  );
}

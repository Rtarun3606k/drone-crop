"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";

export default function AboutPage() {
  const t = useTranslations("about");

  return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-3xl w-full bg-gray-900 rounded-xl shadow-lg p-8 border border-green-500 flex flex-col">
        <h1 className="text-4xl font-bold text-green-400 mb-4 text-center">
          {t("title")}
        </h1>
        <p className="text-gray-200 text-lg mb-6 text-center">
          <span className="font-semibold text-green-400">
            {t("mission_title")}
          </span>
          <br />
          {t("mission_description")}
        </p>
        <p className="text-gray-200 text-lg mb-6 text-center">
          <span className="font-semibold text-green-400">{t("why_title")}</span>
          <br />
          {t("why_description")}
        </p>
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-green-400 mb-2 text-center">
            {t("offer_title")}
          </h2>
          <ul className="list-disc list-inside text-gray-100 space-y-2">
            <li>{t("offer_1")}</li>
            <li>{t("offer_2")}</li>
            <li>{t("offer_3")}</li>
            <li>{t("offer_4")}</li>
          </ul>
        </div>
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

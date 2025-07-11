import { auth } from "@/app/auth";
import { Link, redirect } from "@/i18n/routing";
import React from "react";
import { FiUpload, FiList, FiGrid } from "react-icons/fi";
import { useTranslations } from "next-intl";
import { getTranslations } from "next-intl/server";

const page = async ({ params }) => {
  const session = await auth();
  const t = await getTranslations("dashboard");

  if (!session?.user) {
    return (
      <>
        <Link href={"/login"}> Login to access this page </Link>
      </>
    );
  }

  // This is the main dashboard page for users
  return (
    <div className="min-h-screen bg-[#0A0A0A]">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-white mb-4">{t("welcome")}</h1>
          <p className="text-lg text-gray-300 mb-2">{t("navigation")}</p>
          <p className="text-gray-300">
            {t("questions")}
            <Link href="/contact" className="text-green-500 hover:text-green-400 hover:underline ml-1 font-medium">
              {t("contact")}
            </Link>
            .
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Link
            href="/dashboard/upload"
            className="group flex flex-col items-center border-2 border-dashed border-gray-600 rounded-xl p-8 hover:border-green-500 hover:bg-gray-900 hover:shadow-lg transition-all duration-300 transform hover:scale-105"
          >
            <div className="flex items-center justify-center w-16 h-16 bg-green-900 rounded-full mb-4 group-hover:bg-green-800 transition-colors duration-300">
              <FiUpload className="text-2xl text-green-400 group-hover:text-green-300" />
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold text-white mb-2 group-hover:text-green-300">
                {t("upload_title")}
              </p>
              <p className="text-sm text-gray-400 group-hover:text-gray-300">
                {t("upload_desc")}
              </p>
            </div>
          </Link>

          <Link
            href="/dashboard/batches"
            className="group flex flex-col items-center border-2 border-dashed border-gray-600 rounded-xl p-8 hover:border-green-500 hover:bg-gray-900 hover:shadow-lg transition-all duration-300 transform hover:scale-105"
          >
            <div className="flex items-center justify-center w-16 h-16 bg-green-900 rounded-full mb-4 group-hover:bg-green-800 transition-colors duration-300">
              <FiList className="text-2xl text-green-400 group-hover:text-green-300" />
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold text-white mb-2 group-hover:text-green-300">
                {t("batches_title")}
              </p>
              <p className="text-sm text-gray-400 group-hover:text-gray-300">
                {t("batches_desc")}
              </p>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default page;

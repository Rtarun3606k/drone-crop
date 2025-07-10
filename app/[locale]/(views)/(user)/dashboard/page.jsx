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
    <>
      <div className="mx-auto p-4 flex flex-col items-center font-stretch-semi-expanded text-lg text-center">
        <p className="font-extrabold text-2xl">{t("welcome")}</p>
        <p>{t("navigation")}</p>
        <p>
          {t("questions")}
          <Link href="/contact" className="text-blue-500 hover:underline">
            {t("contact")}
          </Link>
          .
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 px-5">
        <Link
          href="/dashboard/upload"
          className="flex justify-center items-center border-2 border-dashed border-gray-300 rounded-lg p-6 gap-3 hover:border-green-600 hover:bg-green-700 hover:text-black transition-colors duration-300"
        >
          <FiUpload className="text-2xl hover:text-black" />
          <div>
            <p className="text-lg font-semibold hover:text-black">
              {t("upload_title")}
            </p>
            <p className="text-sm text-gray-400 hover:text-black">
              {t("upload_desc")}
            </p>
          </div>
        </Link>

        <Link
          href="/dashboard/batches"
          className="flex justify-center items-center border-2 border-dashed border-gray-300 rounded-lg p-6 gap-3 hover:border-green-600 hover:bg-green-700 hover:text-black transition-colors duration-300"
        >
          <FiList className="text-2xl hover:text-black" />
          <div>
            <p className="text-lg font-semibold hover:text-black">
              {t("batches_title")}
            </p>
            <p className="text-sm text-gray-400 hover:text-black">
              {t("batches_desc")}
            </p>
          </div>
        </Link>
      </div>
    </>
  );
};

export default page;

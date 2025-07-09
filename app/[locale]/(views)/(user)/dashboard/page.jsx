import { auth } from "@/app/auth";
import { Link, redirect } from "@/i18n/routing";
import React from "react";
import { FiUpload, FiList, FiGrid } from "react-icons/fi";

const page = async ({ params }) => {
  const session = await auth();

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
        <p className="font-extrabold text-2xl">Welcome to your dashboard!</p>
        <p>
          Use the navigation menu to access different sections of your
          dashboard.
        </p>
        <p>
          If you have any questions, feel free to{" "}
          <Link href="/contact" className="text-blue-500 hover:underline">
            Contact
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
              Upload Images
            </p>
            <p className="text-sm text-gray-400 hover:text-black">
              Upload new drone images for analysis
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
              View Batches
            </p>
            <p className="text-sm text-gray-400 hover:text-black">
              See all your uploaded batches and analysis results
            </p>
          </div>
        </Link>
      </div>
    </>
  );
};

export default page;

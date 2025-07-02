import { auth } from "@/app/auth";
import { Link, redirect } from "@/i18n/routing";
import React from "react";
import { FiUpload } from "react-icons/fi";

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
        {/* <h1 className="font-extrabold text-4xl">Dashboard</h1> */}
        <p className="font-extrabold text-2xl">Welcome to your dashboard!</p>
        {/* <p>Here you can manage your account, view statistics, and more.</p> */}
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

      <Link
        href="/dashboard/upload"
        className="flex justify-center mt-8 border-2 border-dashed border-gray-300 items-center rounded-lg p-4 gap-2 hover:border-green-600 hover:bg-green-700 hover:text-black transition-colors duration-300 md:w-lg md:h-2xl lg:w-2xl lg:h-3xl xl:w-4xl xl:h-2xl 2xl:w-5xl 2xl:h-5xl m-5"
      >
        <FiUpload className="hover:text-black" />
        <p className="text-lg font-semibold hover:text-black">
          Upload your drone images
        </p>
      </Link>
    </>
  );
};

export default page;

"use client";

import { redirect } from "@/i18n/routing";
import { useRouter } from "next/navigation";
import React, { useEffect } from "react";

const Page = () => {
  const router = useRouter();

  useEffect(() => {
    redirect("/home");
  }, [router]);

  return <div>Redirecting...</div>;
};

export default Page;

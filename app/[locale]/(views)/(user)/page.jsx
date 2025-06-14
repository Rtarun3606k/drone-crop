"use client";

import { redirect } from "@/i18n/routing";
import { useRouter } from "@/i18n/routing";
import React, { useEffect } from "react";

const Page = () => {
  const router = useRouter();

  useEffect(() => {
    router.push("/home");
  }, [router]);

  return <div>Redirecting...</div>;
};

export default Page;

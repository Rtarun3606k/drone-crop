"use client";
import { Link } from "@/i18n/routing";
import { useSession } from "next-auth/react";
import { useTranslations } from "next-intl";

import React from "react";

const page = () => {
  const { data: session } = useSession();

  const t = useTranslations("Home");
  return (
    <div>
      <h1>{t("title")}</h1>
      <Link href="/about">{t("description")}</Link>
    </div>
  );
};
export default page;

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";

import { PrismaClient } from "@prisma/client";

const app = express();
const PORT = process.env.PORT || 4000;

const prisma = new PrismaClient();

app.use(cors("*"));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

export { app, PORT, prisma };

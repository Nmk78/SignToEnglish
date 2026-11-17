import { stopService } from "./service";

export default async function globalTeardown(): Promise<void> {
  await stopService();
}

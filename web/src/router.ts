import { wrap } from "svelte-spa-router/wrap";
import Login from "./routes/Login.svelte";
import Stream from "./routes/Stream.svelte";
import Chat from "./routes/Chat.svelte";
import Settings from "./routes/Settings.svelte";
import NotFound from "./routes/NotFound.svelte";

export const routes = {
  "/login": Login,
  "/": wrap({ component: Stream }),
  "/stream": Stream,
  "/chat": Chat,
  "/settings": Settings,
  "*": NotFound,
};

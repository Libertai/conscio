import { wrap } from "svelte-spa-router/wrap";
import Login from "./routes/Login.svelte";
import Stream from "./routes/Stream.svelte";
import Chat from "./routes/Chat.svelte";
import Projects from "./routes/Projects.svelte";
import Goals from "./routes/Goals.svelte";
import Influences from "./routes/Influences.svelte";
import Memory from "./routes/Memory.svelte";
import Episodes from "./routes/Episodes.svelte";
import Trace from "./routes/Trace.svelte";
import Settings from "./routes/Settings.svelte";
import NotFound from "./routes/NotFound.svelte";

export const routes = {
  "/login": Login,
  "/": wrap({ component: Stream }),
  "/stream": Stream,
  "/chat": Chat,
  "/projects": Projects,
  "/goals": Goals,
  "/influences": Influences,
  "/memory": Memory,
  "/episodes": Episodes,
  "/trace": Trace,
  "/settings": Settings,
  "*": NotFound,
};

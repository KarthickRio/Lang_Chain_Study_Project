import NewTodoForm from "./components/NewTodoForm";

export default function Home() {
  return (
    <section className="w-full max-w-md space-y-4">
      {/* TODO list will appear here after AI wires it up */}
      <NewTodoForm />   {/* AI will complete component & optimistic updates */}
    </section>
  );
}

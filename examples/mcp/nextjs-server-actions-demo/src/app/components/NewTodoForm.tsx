'use client';

import { useFormState } from 'react-dom';
import { createTodo } from '../actions'; // will exist after AI completes

// TODO(sequentialThinking):
// 1. call useFormState with createTodo
// 2. render input + submit button
// 3. optimistic list rendering on success
// 4. show validation errors
export default function NewTodoForm() {
  return (
    <form action={createTodo} className="flex gap-2 items-center">
      {/* input + error display + button go here */}
    </form>
  );
}

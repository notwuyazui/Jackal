// Keep the existing catalog id so the recording-only scenario needs no
// gameplay/schema migration; its visible adapter is now a firefighter.
export { prepareFirefighter as prepare, createFirefighter as create } from '../../src/firefighter.ts';

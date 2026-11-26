// Executable depth-pipeline acceptance evidence.
// Run: npm run acceptance [-- --json]
// The Node replay is an independent attachment/comparator oracle; real GPU
// pixels remain covered by the browser regression entry described in README.
import { runDepthAcceptanceSuite } from '../src/depth-acceptance.js';

const asJson = process.argv.includes('--json');
const evidence = runDepthAcceptanceSuite();

if (asJson) {
  console.log(JSON.stringify(evidence, null, 2));
} else {
  console.log('Depth pipeline acceptance');
  for (const c of evidence.cases) {
    console.log(
      `  ${c.ok ? 'PASS' : 'FAIL'}  ${c.name} [${c.verdict}]` +
        (c.distanceErrorM != null
          ? ` id=${c.pick?.objectId} dErr=${c.distanceErrorM.toExponential(3)} tol=${Number(c.distanceToleranceM).toExponential(3)}`
          : ''),
    );
  }
  console.log(
    `  ${evidence.clipBoundary.ok ? 'PASS' : 'FAIL'}  clip quantization boundary replay (${evidence.clipBoundary.results
      .map((r) => r.correctCode)
      .join(' -> ')})`,
  );
  for (const m of evidence.mutations) {
    console.log(`  ${m.detected ? 'PASS' : 'FAIL'}  mutation detected: ${m.mutation} [${m.actualCodes.join(', ')}]`);
  }
  console.log(`\n${evidence.ok ? 'ACCEPTANCE PASSED' : 'ACCEPTANCE FAILED'}`);
}

if (!evidence.ok) process.exit(1);

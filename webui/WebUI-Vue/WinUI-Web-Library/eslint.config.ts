import { globalIgnores } from 'eslint/config'
import { configureVueProject, defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'
import pluginVue from 'eslint-plugin-vue'
import pluginOxlint from 'eslint-plugin-oxlint'

// The migrated components are authored with plain `<script setup>` (no `lang`),
// consistent with the upstream WinUIonWeb source. Register these languages so
// the TS parser treats a lang-less `<script>` as valid JavaScript.
configureVueProject({ scriptLangs: ['ts', 'js'] })

export default defineConfigWithVueTs(
  {
    name: 'app/files-to-lint',
    files: ['**/*.{vue,ts,mts,tsx}'],
  },

  globalIgnores(['**/dist/**', '**/dist-ssr/**', '**/coverage/**']),

  ...pluginVue.configs['flat/essential'],
  vueTsConfigs.recommended,

  ...pluginOxlint.buildFromOxlintConfigFile('.oxlintrc.json'),

  // Applied last so overrides win over the preset rules above:
  // - Migrated components strip `class`/`style`/`disabled` from $attrs via
  //   underscore-destructuring; honor the `_` prefix for unused vars/args.
  // - WinUI controls intentionally bind BOTH the `:Style` prop (e.g. the
  //   "AccentButtonStyle" name) AND the DOM `:style` on the same element.
  //   eslint-plugin-vue treats these as a duplicate `style` attribute and
  //   even emits a compiler-level parse error, so both rules are relaxed here.
  {
    name: 'app/overrides',
    files: ['**/*.{vue,ts,mts,tsx}'],
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { varsIgnorePattern: '^_', argsIgnorePattern: '^_' }],
      'vue/no-duplicate-attributes': 'off',
      'vue/no-parsing-error': 'off',
    },
  },
)
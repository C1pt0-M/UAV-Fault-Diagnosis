import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const currentDir = dirname(fileURLToPath(import.meta.url))
const sourceRoot = resolve(currentDir, '..')

function readSource(relativePath) {
  return readFileSync(resolve(sourceRoot, relativePath), 'utf-8')
}

describe('frontend style contracts', () => {
  it('uses only defined design tokens in assets analysis panel', () => {
    const source = readSource('modules/assets/components/AssetsAnalysisPanel.vue')

    expect(source).not.toContain('var(--primary)')
    expect(source).not.toContain('var(--bg-soft)')
    expect(source).not.toContain('var(--danger)')
  })

  it('keeps assets workspace controls aligned with global card radius and panel tokens', () => {
    const source = readSource('modules/assets/components/AssetsWorkspacePanel.vue')

    expect(source).not.toContain('border-radius: 18px')
    expect(source).not.toContain('linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%)')
    expect(source).not.toContain('color: var(--primary)')
  })

  it('keeps overview semantic styles in classes instead of inline one-off styles', () => {
    const source = readSource('modules/overview/OverviewPage.vue')

    expect(source).not.toContain('style="color: var(--text-primary); margin-left: 4px;"')
    expect(source).not.toContain('style="color: var(--status-ok); margin-left: 4px;"')
    expect(source).not.toContain('style="font-size: 18px; padding-top: 4px;"')
    expect(source).not.toContain('style="font-size: 11px; color: var(--text-tertiary);"')
    expect(source).not.toContain('style="margin-bottom: 10px;"')
  })

  it('keeps monitoring control panel compact and industrial', () => {
    const source = readSource('modules/monitoring/components/MonitoringControlPanel.vue')

    expect(source).not.toContain('border-radius: 18px')
    expect(source).not.toContain('linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%)')
    expect(source).not.toContain('style="width:')
  })

  it('keeps diagnosis task launcher compact and industrial', () => {
    const source = readSource('modules/diagnosis/components/DiagnosisTaskForm.vue')

    expect(source).not.toContain('border-radius: 18px')
    expect(source).not.toContain('linear-gradient(180deg, #f6fbfc 0%, #eef6f7 100%)')
  })

  it('keeps global density comfortable instead of cramped', () => {
    const mainCss = readSource('styles/main.css')
    const appShell = readSource('layouts/AppShell.vue')

    expect(mainCss).toContain('--space-xl:  24px')
    expect(mainCss).toContain('--space-2xl: 32px')
    expect(mainCss).toContain('font-size: 15px')
    expect(mainCss).toContain('.section-card__body {\n  padding: var(--space-xl);')
    expect(appShell).toContain('height: 56px')
    expect(appShell).toContain('padding: 28px 32px')
  })

  it('keeps collapsed sidebar icons visually centered', () => {
    const appShell = readSource('layouts/AppShell.vue')

    expect(appShell).toContain('.is-collapsed .shell-nav')
    expect(appShell).toContain('align-items: center')
    expect(appShell).toContain('width: 36px')
    expect(appShell).toContain('height: 36px')
    expect(appShell).toContain('gap: 0')
    expect(appShell).not.toContain('.is-collapsed .nav-item.is-active {\n  border-left: 2px solid var(--brand);')
  })

  it('keeps dark theme in readable industrial slate instead of near-black', () => {
    const mainCss = readSource('styles/main.css')

    expect(mainCss).not.toContain('--bg-canvas:       #0f111a')
    expect(mainCss).not.toContain('--bg-panel:        #161925')
    expect(mainCss).not.toContain('--bg-panel-raised: #1f2333')
    expect(mainCss).toContain('--bg-canvas:       #1b2230')
    expect(mainCss).toContain('--bg-panel:        #242c3a')
    expect(mainCss).toContain('--el-bg-color: var(--bg-panel)')
    expect(mainCss).toContain('--el-fill-color-blank: var(--bg-panel)')
  })

  it('keeps dark-sensitive workbench surfaces token based', () => {
    const sources = [
      readSource('modules/assets/components/AssetDetailDialog.vue'),
      readSource('modules/settings/SettingsPage.vue')
    ].join('\n')

    expect(sources).not.toContain('background: #0f1f26')
    expect(sources).not.toContain('color: #f4fbfb')
    expect(sources).not.toContain('background: #0f111a')
    expect(sources).not.toContain('background: #161925')
  })

  it('does not contain real mojibake characters in global UI sources', () => {
    const globalSources = [
      readSource('styles/main.css'),
      readSource('layouts/AppShell.vue')
    ].join('\n')

    expect(globalSources).not.toMatch(/[鈹鏅绯閫鏍艰埅]/)
  })

  it('keeps logged-in workbench modules off the legacy teal visual system', () => {
    const workbenchSources = [
      readSource('modules/ai/components/AIHeroSection.vue'),
      readSource('modules/ai/components/AIContextPanel.vue'),
      readSource('modules/reports/components/ReportsWorkspacePanel.vue'),
      readSource('modules/monitoring/components/MonitoringMetrics.vue'),
      readSource('modules/monitoring/components/MonitoringSessionsPanel.vue')
    ].join('\n')

    expect(workbenchSources).not.toMatch(/#0f766e|#103240|#eef6f7|#f7fbfc|#081f2f|#0d4656|#f4fbfb/)
    expect(workbenchSources).not.toMatch(/border-radius:\s*(18|20)px/)
    expect(workbenchSources).not.toContain('linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%)')
  })

  it('keeps settings and reports layout styles semantic', () => {
    const settingsSource = readSource('modules/settings/SettingsPage.vue')
    const reportsSource = readSource('modules/reports/components/ReportsWorkspacePanel.vue')

    expect(settingsSource).not.toContain('style="margin-top:')
    expect(settingsSource).not.toContain('style="position: relative;"')
    expect(settingsSource).not.toContain('style="display: flex; gap: 8px;"')
    expect(settingsSource).not.toContain('style="flex: 1;"')
    expect(reportsSource).not.toContain('style="width: 220px"')
  })

  it('keeps shared chart colors aligned with design tokens', () => {
    const chartSource = readSource('shared/components/TimeSeriesChart.vue')

    expect(chartSource).not.toContain('#0f766e')
    expect(chartSource).not.toContain('#738791')
    expect(chartSource).not.toContain('#49616d')
    expect(chartSource).toContain('getComputedStyle')
  })
})

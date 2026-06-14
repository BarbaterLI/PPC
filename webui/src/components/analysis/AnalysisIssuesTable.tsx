import {
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
  TableCellLayout,
} from '@fluentui/react-components'
import { StatusBadge } from '@/components/StatusBadge'
import type { StatusType } from '@/components/StatusBadge'

const MODULE_LABELS: Record<string, string> = {
  performance: '性能',
  config: '配置',
  errors: '错误',
  dependency: '依赖',
  network: '网络',
  resource: '资源',
  code_quality: '代码质量',
}

const SEVERITY_MAP: Record<string, StatusType> = {
  critical: 'failed',
  high: 'failed',
  medium: 'warning',
  low: 'pending',
  info: 'completed',
}

const SEVERITY_LABELS: Record<string, string> = {
  critical: '严重',
  high: '高',
  medium: '中',
  low: '低',
  info: '信息',
}

interface AnalysisIssueItem {
  analyzer: string
  severity: string
  message: string
  suggestion?: string
}

const issueColumns: TableColumnDefinition<AnalysisIssueItem>[] = [
  createTableColumn({
    columnId: 'analyzer',
    renderHeaderCell: () => '分析器',
    renderCell: (item) => <TableCellLayout>{MODULE_LABELS[item.analyzer] || item.analyzer}</TableCellLayout>,
  }),
  createTableColumn({
    columnId: 'severity',
    renderHeaderCell: () => '严重程度',
    renderCell: (item) => (
      <TableCellLayout>
        <StatusBadge status={SEVERITY_MAP[item.severity] || 'pending'} label={SEVERITY_LABELS[item.severity] || item.severity} />
      </TableCellLayout>
    ),
  }),
  createTableColumn({
    columnId: 'message',
    renderHeaderCell: () => '消息',
    renderCell: (item) => <TableCellLayout>{item.message}</TableCellLayout>,
  }),
  createTableColumn({
    columnId: 'suggestion',
    renderHeaderCell: () => '建议',
    renderCell: (item) => <TableCellLayout>{item.suggestion || '—'}</TableCellLayout>,
  }),
]

interface AnalysisIssuesTableProps {
  issues: AnalysisIssueItem[]
}

export function AnalysisIssuesTable({
  issues,
}: AnalysisIssuesTableProps) {
  return (
    <DataGrid items={issues} columns={issueColumns} sortable>
      <DataGridHeader>
        <DataGridRow>
          {({ renderHeaderCell }) => (
            <DataGridCell>{renderHeaderCell()}</DataGridCell>
          )}
        </DataGridRow>
      </DataGridHeader>
      <DataGridBody<AnalysisIssueItem>>
        {({ item, rowId }) => (
          <DataGridRow key={rowId}>
            {({ renderCell }) => (
              <DataGridCell>{renderCell(item)}</DataGridCell>
            )}
          </DataGridRow>
        )}
      </DataGridBody>
    </DataGrid>
  )
}

import {
  makeStyles,
  tokens,
  shorthands,
  Text,
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
  TableCellLayout,
  Spinner,
} from '@fluentui/react-components'

const MODULE_LABELS: Record<string, string> = {
  performance: '性能',
  config: '配置',
  errors: '错误',
  dependency: '依赖',
  network: '网络',
  resource: '资源',
  code_quality: '代码质量',
}

const useStyles = makeStyles({
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '200px',
  },
  emptyHistory: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground2,
    textAlign: 'center',
    ...shorthands.padding(tokens.spacingVerticalXL, 0),
  },
  historyRow: {
    cursor: 'pointer',
  },
})

function getScoreColor(score: number) {
  if (score >= 80) return tokens.colorPaletteGreenForeground1
  if (score >= 60) return tokens.colorPaletteYellowForeground1
  return tokens.colorPaletteRedForeground1
}

interface AnalysisHistoryItem {
  id: string
  taskId?: string
  date: string
  score: number
  analyzers: string[]
  summary?: string
}

const historyColumns: TableColumnDefinition<AnalysisHistoryItem>[] = [
  createTableColumn({
    columnId: 'date',
    renderHeaderCell: () => '日期',
    renderCell: (item) => <TableCellLayout>{new Date(item.date).toLocaleString('zh-CN')}</TableCellLayout>,
  }),
  createTableColumn({
    columnId: 'score',
    renderHeaderCell: () => '评分',
    renderCell: (item) => (
      <TableCellLayout>
        <span style={{ color: getScoreColor(item.score), fontWeight: tokens.fontWeightSemibold }}>
          {item.score}
        </span>
      </TableCellLayout>
    ),
  }),
  createTableColumn({
    columnId: 'analyzers',
    renderHeaderCell: () => '分析模块',
    renderCell: (item) => <TableCellLayout>{item.analyzers?.map((a) => MODULE_LABELS[a] || a).join(', ') || '—'}</TableCellLayout>,
  }),
]

interface AnalysisHistoryTableProps {
  historyData: AnalysisHistoryItem[] | null
  historyLoading: boolean
  onViewHistoryItem: (item: AnalysisHistoryItem) => void
}

export function AnalysisHistoryTable({
  historyData,
  historyLoading,
  onViewHistoryItem,
}: AnalysisHistoryTableProps) {
  const styles = useStyles()

  if (historyLoading) {
    return (
      <div className={styles.loadingContainer}>
        <Spinner label="加载历史记录..." />
      </div>
    )
  }

  if (!historyData || historyData.length === 0) {
    return (
      <div className={styles.emptyHistory}>
        <Text>暂无历史分析记录</Text>
      </div>
    )
  }

  return (
    <DataGrid items={historyData} columns={historyColumns} sortable>
      <DataGridHeader>
        <DataGridRow>
          {({ renderHeaderCell }) => (
            <DataGridCell>{renderHeaderCell()}</DataGridCell>
          )}
        </DataGridRow>
      </DataGridHeader>
      <DataGridBody<AnalysisHistoryItem>>
        {({ item, rowId }) => (
          <DataGridRow
            key={rowId}
            className={styles.historyRow}
            onClick={() => onViewHistoryItem(item)}
          >
            {({ renderCell }) => (
              <DataGridCell>{renderCell(item)}</DataGridCell>
            )}
          </DataGridRow>
        )}
      </DataGridBody>
    </DataGrid>
  )
}

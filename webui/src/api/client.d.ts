import type { ApiResponse, SystemStatus, PPC10Config, ConfigSection, ConfigUpdatePayload, ConfigExportData, Task, TaskProgress, ConvertTaskParams, AnalysisResult, Extension, ExtensionWebUIConfig, FanqieStatus, FanqieConfig, FanqieInstallParams, FanqieLaunchServerParams, VoiceGroup, SplitParams, MergeParams, PreviewParams, HealthReport, ClusterStatus, NodeInfo, ClusterMetricsResponse, TaskAssignment } from '../types';
declare class ApiError extends Error {
    status: number;
    code?: string;
    constructor(message: string, status: number, code?: string);
}
declare function request<T>(url: string, options?: RequestInit, signal?: AbortSignal): Promise<T>;
declare function requestBlob(url: string, options?: RequestInit, signal?: AbortSignal): Promise<Blob>;
declare function createSSEConnection(url: string, onMessage: (data: TaskProgress) => void, onError?: (error: Event) => void): EventSource;
export { ApiError, request, requestBlob, createSSEConnection };
export declare const systemApi: {
    getStatus: () => Promise<ApiResponse<SystemStatus>>;
    check: () => Promise<ApiResponse<HealthReport>>;
    getVoices: () => Promise<ApiResponse<VoiceGroup[]>>;
};
export declare const configApi: {
    getAll: () => Promise<ApiResponse<PPC10Config[]>>;
    getByKey: (key: string) => Promise<ApiResponse<ConfigSection>>;
    update: (payload: ConfigUpdatePayload) => Promise<ApiResponse<void>>;
    batchUpdate: (payloads: Array<{
        key: string;
        value: string;
    }>) => Promise<ApiResponse<{
        failed?: Array<{
            key: string;
        }>;
    }>>;
    reset: (preset?: string) => Promise<ApiResponse<void>>;
    export: () => Promise<ApiResponse<ConfigExportData>>;
    exportFile: () => Promise<Blob>;
    import: (data: ConfigExportData) => Promise<ApiResponse<void>>;
    importFile: (file: File) => Promise<ApiResponse<void>>;
};
export declare const taskApi: {
    createConvert: (params: ConvertTaskParams) => Promise<ApiResponse<{
        task_id: string;
    }>>;
    list: () => Promise<ApiResponse<Task[]>>;
    getById: (id: string) => Promise<ApiResponse<Task>>;
    streamProgress: (id: string) => string;
    cancel: (id: string) => Promise<ApiResponse<void>>;
};
export declare const analyzeApi: {
    run: (modules?: string[]) => Promise<ApiResponse<AnalysisResult>>;
    getHistory: () => Promise<ApiResponse<AnalysisResult[]>>;
};
export declare const operationApi: {
    split: (params: SplitParams) => Promise<ApiResponse<{
        output_dir: string;
        file_count: number;
    }>>;
    merge: (params: MergeParams) => Promise<ApiResponse<{
        output_path: string;
    }>>;
    preview: (params: PreviewParams) => Promise<ApiResponse<{
        audio_url: string;
    }>>;
};
export declare const extensionApi: {
    list: () => Promise<ApiResponse<Extension[]>>;
    getWebUIConfigs: () => Promise<ApiResponse<ExtensionWebUIConfig[]>>;
    install: (file: File) => Promise<ApiResponse<Extension>>;
    uninstall: (name: string) => Promise<ApiResponse<void>>;
    getByName: (name: string) => Promise<ApiResponse<Extension>>;
    enable: (name: string) => Promise<ApiResponse<void>>;
    disable: (name: string) => Promise<ApiResponse<void>>;
};
export declare const fanqieApi: {
    getStatus: () => Promise<ApiResponse<FanqieStatus>>;
    install: (params?: FanqieInstallParams) => Promise<ApiResponse<{
        version: string;
    }>>;
    launchServer: (params?: FanqieLaunchServerParams) => Promise<ApiResponse<{
        host: string;
        port: number;
    }>>;
    stopServer: () => Promise<ApiResponse<void>>;
    getConfig: () => Promise<ApiResponse<FanqieConfig>>;
    updateConfig: (config: Partial<FanqieConfig>) => Promise<ApiResponse<void>>;
    uninstall: () => Promise<ApiResponse<void>>;
};
export declare const distributedApi: {
    getStatus: () => Promise<ApiResponse<ClusterStatus>>;
    getNodes: () => Promise<ApiResponse<NodeInfo[]>>;
    addNode: (host: string, port: number, maxConcurrency: number) => Promise<ApiResponse<NodeInfo>>;
    removeNode: (nodeId: string) => Promise<ApiResponse<void>>;
    drainNode: (nodeId: string) => Promise<ApiResponse<NodeInfo>>;
    activateNode: (nodeId: string) => Promise<ApiResponse<NodeInfo>>;
    getMetrics: () => Promise<ApiResponse<ClusterMetricsResponse>>;
    getTasks: () => Promise<ApiResponse<TaskAssignment[]>>;
    startScheduler: (strategy?: string, localExecution?: boolean) => Promise<ApiResponse<{
        status: string;
    }>>;
    stopScheduler: () => Promise<ApiResponse<{
        status: string;
    }>>;
    startNodeService: (host?: string, port?: number, maxConcurrency?: number) => Promise<ApiResponse<{
        status: string;
    }>>;
    stopNodeService: () => Promise<ApiResponse<{
        status: string;
    }>>;
};

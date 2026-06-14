export interface PipelineStep {
    name: string;
    step_type: string;
    depends_on: string[];
    params: Record<string, any>;
    retry_count: number;
    timeout_seconds: number | null;
    on_failure: string;
}
export interface PipelineDefinition {
    name: string;
    description: string;
    steps: PipelineStep[];
    variables: Record<string, string>;
    enabled: boolean;
}
export interface PipelineInfo {
    id: string;
    name: string;
    description: string;
    step_count: number;
    source: string;
}
export interface StepResult {
    step_name: string;
    status: string;
    output_path: string | null;
    error: string | null;
    duration_seconds: number;
}
export interface PipelineRun {
    run_id: string;
    pipeline_name: string;
    status: string;
    step_results: Record<string, StepResult>;
    started_at: string | null;
    completed_at: string | null;
    duration_seconds: number;
}
export interface StepType {
    name: string;
    input_type: string;
    output_type: string;
}

type SSEHookOptions = {
    onMessage?: (event: MessageEvent) => void;
    onError?: (error: Event) => void;
    onOpen?: () => void;
};
export declare function useSSE<T = any>(options?: SSEHookOptions): {
    data: T | null;
    error: Error | null;
    isConnected: boolean;
    connect: (url: string, eventListeners?: Record<string, (event: MessageEvent) => void>) => EventSource | null;
    close: () => void;
    eventSource: EventSource | null;
};
export {};

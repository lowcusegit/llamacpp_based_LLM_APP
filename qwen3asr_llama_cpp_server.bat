llama-server -m C:\GGUF\mradermacher_Qwen3-ASR-0.6B-GGUF\Qwen3-ASR-0.6B.Q4_K_M.gguf ^
--mmproj C:\GGUF\mradermacher_Qwen3-ASR-0.6B-GGUF\Qwen3-ASR-0.6B.mmproj-Q8_0.gguf ^
-t 4 -n 256 --ctx-size 4096 --port 8080
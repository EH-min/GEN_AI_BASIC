<script lang="ts">
	import { tick } from 'svelte';

	let options = [
		{ value: 'openai', label: 'openai/DallE-2' },
		{ value: 'imagen', label: 'google/imagen-3' },
		{ value: 'sd', label: 'stability-ai/stable-diffusion-3.5-large' },
		{ value: 'flux', label: 'black-forest-labs/flux-pro' },
		{ value: 'recraft', label: 'recraft-ai/recraft-v3' }
	];

	let selected = options[0].value;
	let url = '';
	let loading = false;
	let promptInput!: HTMLTextAreaElement;
	let promptMessage = '';

	async function sendPrompt() {
		if (!promptMessage.trim()) {
			promptInput.focus();
			return;
		}
    try {
      // 1. Backend 비동기 통신 프롬프트 전달
      const res = await fetch("http://localhost:8000/api/image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: promptMessage
        }),
      });
      // 2. 생성된 이미지 URL 받기
      const { url: image_url } = await res.json();
      // 3.이미지 출력
      url = image_url;
      await tick();
    } catch (e) {
      console.error("prompt error", e);
    } finally {
      await tick();
      promptInput.focus();
    }
	}
</script>
{#if loading}
	<div class="overlay">
		<div class="spinner"></div>
	</div>
{/if}

<div class="wrap">
	<aside>
		<div class="side_wrap">
			<div class="title_wrap">
				<div class="title_box">
					<div>
						<div><span>이미지 생성</span></div>
					</div>
				</div>

				<div class="select_wrap">
					<select id="model" class="select_model" bind:value={selected}>
						{#each options as option}
							<option value={option.value}>{option.label}</option>
						{/each}
					</select>
					<svg
						class="arrow"
						width="16"
						height="16"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					>
						<polyline points="6 9 12 15 18 9"></polyline>
					</svg>
				</div>
			</div>
			<div class="common_panel gen_box">
				<form class="prompt_box" on:submit|preventDefault={sendPrompt}>
					<textarea
						placeholder="프롬프트를 작성해주세요."
						class="prompt_input"
						bind:value={promptMessage}
						bind:this={promptInput}
					></textarea>
					<div class="msg_warning">"???" 도구는 실수를 할 수 있으니 다시 한번 확인하세요.</div>
					<button class="btn_gen" type="submit">Generate</button>
				</form>
			</div>
		</div>
	</aside>
	<section>
		<div class="content_wrap">
			<div class="common_panel">
				<div class="image_box" style="background-image: url({url})"></div>
			</div>
		</div>
	</section>
</div>
<style>
	.wrap {
		height: calc(100vh - 6.5rem);
		min-width: 134rem;
		display: flex;
		justify-content: center;
		align-items: center;
		box-sizing: border-box;
		padding: 1rem;
		overflow: hidden;
	}

	aside {
		width: 55rem;
		margin-right: 1.5rem;
	}

	.side_wrap {
		display: flex;
		flex-direction: column;
	}

	.common_panel {
		border-radius: 0.5rem;
		background-color: white;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		box-shadow:
			0 0.4rem 1rem 0 rgba(0, 0, 0, 0.2),
			0 0.4rem 2rem 0 rgba(0, 0, 0, 0.19);
		margin-bottom: 1.5rem;
	}

	.gen_box {
		border: 0.3rem solid white;
	}

	.gen_box:focus-within {
		border: 0.3rem solid red;
	}

	.title_wrap {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 2rem; /* 가운데 여백 */
		margin-bottom: 1.5rem;
	}

	.select_wrap {
		border-radius: 0.5rem;
		flex: 2;
		position: relative;
		display: inline-block;
		box-shadow:
			0 0.4rem 1rem 0 rgba(0, 0, 0, 0.2),
			0 0.4rem 2rem 0 rgba(0, 0, 0, 0.19);
	}

	.title_box {
		background-color: #2f3138;
		color: white;
		font-weight: bold;
		padding: 0.7rem 1.5rem;
		font-size: 1.7rem;
		border-radius: 0.5rem;
		height: 5rem;
		flex: 1;
		box-shadow:
			0 0.4rem 1rem 0 rgba(0, 0, 0, 0.2),
			0 0.4rem 2rem 0 rgba(0, 0, 0, 0.19);
	}

	.title_box span {
		display: inline-block;
		width: 100%;
		text-align: center;
		line-height: 3.5rem;
	}

	.select_model {
		width: 100%;
		outline: none;
		padding: 0.7rem 1.5rem;
		font-size: 1.7rem;
		height: 5rem;
		cursor: pointer;
		border: 0.3rem solid blue;
		border-radius: 0.5rem;
		appearance: none; /* 기본 화살표 제거 */
		-webkit-appearance: none;
		-moz-appearance: none;
		background-color: white;
	}

	.select_model:focus {
		border: 0.3rem solid blue;
	}

	.arrow {
		position: absolute;
		top: 50%;
		right: 10px;
		pointer-events: none; /* 클릭 안 되도록 */
		transform: translateY(-50%) rotate(0deg);
		transition: transform 0.2s ease;
	}

	.prompt_box {
		width: 100%;
		position: relative;
	}

	.prompt_box {
		height: 32rem;
	}

	.prompt_box > textarea {
		width: 100%;
		outline: none;
		font-size: 2rem;
		font-family: inherit;
		border: none;
		resize: none;
		padding: 1.5rem;
		line-height: 1.75rem;
		height: 25rem;
	}

	.msg_warning {
		position: absolute;
		font-size: 1.2rem;
		padding: 0 1.5rem;
		color: #dc143c;
		bottom: 1.5rem;
	}

	.btn_gen {
		display: inline-block;
		font-size: 1.7rem;
		font-weight: bold;
		color: white;
		background: black;
		padding: 1rem 2.6rem;
		border-radius: 4rem;
		cursor: pointer;
		position: absolute;
		bottom: 1.5rem;
		right: 2rem;
		z-index: 1;
		outline: none;
		border: none;
		transition: all 0.3s ease;
	}

	.btn_gen:hover {
		box-shadow:
			0 0 1rem #fff,
			0.5rem 0 2rem red,
			-0.5rem 0 2rem blue;
	}

	.content_wrap {
		display: flex;

		flex-direction: column;
	}

	.image_box {
		border-radius: 0.5rem;
		max-width: 45rem;
		max-height: 45rem;
		width: 100%;
		min-width: 38.5rem;
		min-height: 38.5rem;
		border: 0.3rem solid black;
		text-align: center;
		line-height: 38.5rem;
		font-size: 2rem;
		background-size: cover;
		background-position: center;
		background-repeat: no-repeat;
	}

	.overlay {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background-color: rgba(0, 0, 0, 0.4); /* 반투명 */
		display: flex;
		justify-content: center;
		align-items: center;
		z-index: 9999; /* 최상위 */
	}

	.spinner {
		border: 6px solid #f3f3f3;
		border-top: 6px solid blue;
		border-radius: 50%;
		width: 60px;
		height: 60px;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		0% {
			transform: rotate(0deg);
		}

		100% {
			transform: rotate(360deg);
		}
	}
</style>

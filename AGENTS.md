# Repository Guidelines

## 作業の許可範囲

このディレクトリ外に影響を及ぼさない限り、ファイルの作成・変更、および `uv`・`python` の実行は、追加確認なしで自由に行ってよい。

相対パスの基準はこのディレクトリとし、出力ファイル、一時ファイル、仮想環境、キャッシュはこのディレクトリ内に配置する。例として、仮想環境には `.venv/`、uv のキャッシュには `.uv-cache/` を使用する。

## Project Structure & Module Organization

- `data/`: データを配置するディレクトリ。
- `data/test/`: 既存のサブディレクトリ。用途と内容を確認してから変更する。
- 新しいコードやテストは、目的が分かる名前で整理する。既存データの意味や形式を推測で変更しない。

## Build, Test, and Development Commands

現時点でルートにビルド設定やテスト設定は確認されていない。実行対象の設定を確認してからコマンドを選ぶ。

PowerShell でローカル環境を準備する例:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv venv .venv
.\.venv\Scripts\python.exe --version
```

依存関係を追加した場合は、再現できるよう設定ファイルと実行手順を残す。

## Coding Style & Naming Conventions

既存ファイルのスタイルに合わせる。Python を追加する場合はインデントをスペース4個とし、関数・変数・モジュール名には `snake_case` を使う。フォーマッターやリンターを導入した場合は設定と実行方法を明記する。

## Testing Guidelines

テストフレームワークやカバレッジ基準は未設定。動作変更時は変更内容に応じた確認を行い、実行したコマンドと結果を報告する。Python の自動テストを追加する場合は `tests/test_*.py` を基本とする。

## Commit & Pull Request Guidelines

Git 履歴は取得できていないため、既存のコミット規約は未確認。コミットには変更目的が分かる短い件名を付ける。プルリクエストには変更内容、理由、確認方法、関連する課題を記載する。認証情報や不要な生成物は含めない。
## シェルの実行経路

この環境ではサンドボックス経由のシェル起動が Windows エラー1385（`CreateProcessWithLogonW failed`）で失敗するため、シェルコマンドは常にサンドボックス外の実行経路を使用する。`exec_command` では `sandbox_permissions: "require_escalated"` を指定し、`shell: "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"`、`login: false` を使用する。サンドボックス経由での起動を毎回試す必要はない。

この実行経路を使用する場合も、作業の影響はこのディレクトリ内に限定し、実行環境が要求する承認手続きには従う。

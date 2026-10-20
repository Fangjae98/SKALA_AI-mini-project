"""분산에너지 운영 에이전트 — 두 번째(미사용) MCP 서버 ("docs").

agent는 이 서버의 어떤 capability도 직접 호출하지 않는다. 여러 MCP 서버가 동시에
연결돼 있을 때, agent가 필요한 resource를 찾기 위해 "어느 서버가 그 resource를
제공하는지" 먼저 조회(list_resource_templates)하고서야 요청해야 하는 상황을
시뮬레이션하기 위한 decoy(미끼) 서버다.

[Base 대비 변경] 문서 내용만 분산에너지 관련 참고 문서(계통연계 기준, 요금제)로 교체.
"""
from fastmcp import FastMCP

mcp = FastMCP(name="der-ops-agent-docs")

DOC_SECTIONS = {
    "grid-code": "# 참고 문서: 계통연계 기준\n\n분산전원 연계 시 전압·역률 유지 범위와 보호 협조 기준을 설명합니다.",
    "tariff": "# 참고 문서: 전기요금제\n\n계약전력, 기본요금, 시간대별 전력량 요금 구조를 설명합니다.",
}


@mcp.resource("docs://reference/{section}")
def get_reference_doc(section: str) -> str:
    """분산에너지 참고 문서(section별 정적 문서)를 반환합니다. (agent는 사용하지 않음)"""
    if section not in DOC_SECTIONS:
        return f"# Doc Not Found\n\nNo reference for section: {section}"
    return DOC_SECTIONS[section]


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:   # 구버전 fastmcp 호환
        mcp.run()

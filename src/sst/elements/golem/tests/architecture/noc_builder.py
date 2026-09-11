import os
import sst
from typing import Dict, List, Optional, Tuple, Union

Endpoint = Union[Tuple[object, str], Tuple[object, str, str]]
ENABLE_ALL_STATS = int(os.getenv("GOLEM_SST_ENABLE_ALL_STATS", "1")) != 0


class MeshNoCBuilder:
    """Helper that instantiates a Merlin mesh NoC and provides local attachment helpers."""

    def __init__(
        self,
        dim_x: int = 2,
        dim_y: int = 2,
        *,
        local_ports: int = 2,
        router_prefix: str = "rtr",
        link_bw: str = "25GB/s",
        xbar_bw: str = "25GB/s",
        flit_size: str = "72B",
        directional_link_latency: str = "1ns",
        local_link_latency: str = "1ns",
        input_buf_size: str = "2KB",
        output_buf_size: str = "2KB",
        num_vns: int = 3,
        vn_priority_order: str = "",
        vn_starvation_vn: int = -1,
        vn_max_starvation_cycles: int = 0,
        width: str = "1x1",
        verify_topology: bool = True,
        inter_router_no_cut: bool = True,
        local_no_cut: bool = True,
        debug: int = 0,
    ) -> None:
        if dim_x <= 0 or dim_y <= 0:
            raise ValueError("Mesh dimensions must be positive")
        if local_ports <= 0:
            raise ValueError("Each router must expose at least one local port")

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.local_ports = local_ports
        self.router_prefix = router_prefix
        self.directional_link_latency = directional_link_latency
        self.local_link_latency = local_link_latency
        self.verify_topology = verify_topology
        self.inter_router_no_cut = inter_router_no_cut
        self.local_no_cut = local_no_cut
        self.num_nodes = dim_x * dim_y
        self.local_port_base = 4  # Merlin hr_router reserves ports 0-3 for cardinal links.

        self._router_entries: List[Dict[str, object]] = []
        self._local_usage: List[int] = [0 for _ in range(self.num_nodes)]
        self._local_link_index: int = 0

        base_router_params = {
            "num_ports": str(4 + local_ports),
            "link_bw": link_bw,
            "xbar_bw": xbar_bw,
            "flit_size": flit_size,
            "input_latency": directional_link_latency,
            "output_latency": directional_link_latency,
            "input_buf_size": input_buf_size,
            "output_buf_size": output_buf_size,
            "num_vns": num_vns,
            "vn_priority_order": vn_priority_order,
            "vn_starvation_vn": str(vn_starvation_vn),
            "vn_max_starvation_cycles": str(vn_max_starvation_cycles),
            "debug": str(debug),  # Merlin 路由器只支持 debug (0/1)
        }
        self.router_params = base_router_params
        self.topology_params = {
            "shape": f"{dim_x}x{dim_y}",
            "width": width,
            "local_ports": str(local_ports),
        }

    @property
    def routers(self) -> List[sst.Component]:
        return [entry["component"] for entry in self._router_entries]

    def build(self) -> "MeshNoCBuilder":
        self._instantiate_routers()
        self._wire_cardinal_links()
        if self.verify_topology:
            self.print_topology_summary()
        return self

    def get_router(self, index: int) -> sst.Component:
        try:
            return self._router_entries[index]["component"]  # type: ignore[index]
        except IndexError as exc:
            raise IndexError(f"Router index {index} is out of range for mesh size {self.dim_x}x{self.dim_y}") from exc

    def available_local_ports(self, index: int) -> int:
        total = self.local_ports
        used = self._local_usage[index]
        return total - used

    def attach_local(
        self,
        router_index: int,
        endpoint: Endpoint,
        *,
        link_name: Optional[str] = None,
        endpoint_latency: Optional[str] = None,
        router_latency: Optional[str] = None,
        no_cut: Optional[bool] = None,
    ) -> sst.Link:
        router_entry = self._router_entries[router_index]
        if self._local_usage[router_index] >= self.local_ports:
            raise RuntimeError(
                f"Router {router_index} has no remaining local ports (requested endpoint {endpoint})."
            )

        component, port, ep_latency = self._normalize_endpoint(endpoint)
        latency_component = endpoint_latency or ep_latency or self.local_link_latency
        latency_router = router_latency or self.local_link_latency

        port_offset = self._local_usage[router_index]
        router_port_number = self.local_port_base + port_offset
        router_port = f"port{router_port_number}"
        self._local_usage[router_index] += 1

        if link_name is None:
            link_name = f"link_{router_entry['name']}_{router_port_number}_{self._local_link_index}"
        self._local_link_index += 1

        link = sst.Link(link_name)
        link.connect((component, port, latency_component), (router_entry["component"], router_port, latency_router))  # type: ignore[index]
        should_no_cut = self.local_no_cut if no_cut is None else no_cut
        if should_no_cut:
            link.setNoCut()
        return link

    def print_topology_summary(self) -> None:
        print(
            f"\n[NoC] Mesh {self.dim_x}x{self.dim_y} connectivity (E = link present, . = boundary)"
        )
        header = "     ID  (x,y)   +X  -X  +Y  -Y  LocalPorts"
        print(header)
        print("     " + "-" * (len(header) - 4))
        for node_id in range(self.num_nodes):
            x = node_id % self.dim_x
            y = node_id // self.dim_x
            has_px = "E" if x < self.dim_x - 1 else "."
            has_mx = "E" if x > 0 else "."
            has_py = "E" if y < self.dim_y - 1 else "."
            has_my = "E" if y > 0 else "."
            remaining = self.available_local_ports(node_id)
            print(f"     {node_id:2d}  ({x},{y})   {has_px}   {has_mx}   {has_py}   {has_my}      {self.local_ports} ({remaining} free)")
        internal_links_expected = (self.dim_x - 1) * self.dim_y + (self.dim_y - 1) * self.dim_x
        print(f"[NoC] Expected internal (mesh) link count: {internal_links_expected}\n")

    def _instantiate_routers(self) -> None:
        if self._router_entries:
            return

        for router_id in range(self.num_nodes):
            router_name = f"{self.router_prefix}_{router_id}"
            router = sst.Component(router_name, "merlin.hr_router")
            router.addParams({"id": str(router_id), **self.router_params})
            if ENABLE_ALL_STATS:
                router.enableAllStatistics({"type": "sst.AccumulatorStatistic"})

            topo = router.setSubComponent("topology", "merlin.mesh")
            topo.addParams(self.topology_params)

            self._router_entries.append({"component": router, "name": router_name})

    def _wire_cardinal_links(self) -> None:
        P_PX, P_NX, P_PY, P_NY = 0, 1, 2, 3
        for node_id in range(self.num_nodes):
            x = node_id % self.dim_x
            y = node_id // self.dim_x

            if x < self.dim_x - 1:
                east = node_id + 1
                link = sst.Link(f"link_h_{node_id}_{east}")
                link.connect(
                    (self._router_entries[node_id]["component"], f"port{P_PX}", self.directional_link_latency),
                    (self._router_entries[east]["component"], f"port{P_NX}", self.directional_link_latency),
                )
                if self.inter_router_no_cut:
                    link.setNoCut()

            if y < self.dim_y - 1:
                south = node_id + self.dim_x
                link = sst.Link(f"link_v_{node_id}_{south}")
                link.connect(
                    (self._router_entries[node_id]["component"], f"port{P_PY}", self.directional_link_latency),
                    (self._router_entries[south]["component"], f"port{P_NY}", self.directional_link_latency),
                )
                if self.inter_router_no_cut:
                    link.setNoCut()

    @staticmethod
    def _normalize_endpoint(endpoint: Endpoint) -> Tuple[object, str, Optional[str]]:
        if not isinstance(endpoint, tuple):
            raise TypeError(f"Endpoint must be a tuple, received {type(endpoint)}")
        if len(endpoint) == 2:
            component, port = endpoint
            return component, port, None
        if len(endpoint) == 3:
            component, port, latency = endpoint
            return component, port, latency
        raise ValueError(
            "Endpoint tuples must be of the form (component, port) or (component, port, latency)."
        )

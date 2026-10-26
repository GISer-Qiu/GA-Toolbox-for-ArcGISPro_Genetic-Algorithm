# -*- coding: utf-8 -*-
"""
GA_Route_Optimizer.pyt
基于遗传算法的路网游览路径优化 Python 工具箱

相较  GA 8.5.pyt，做了以下调整：
  - 参数界面更贴近需求：输入点、编码字段、路网、算法参数、输出数据库
  - 在参数页面增加编码字段唯一性、取值范围、输出数据库等校验
  - 遗传算法部分避免重复计算适应度，精英保留更直观
  - 输出结果保留“最优路径折线”和“POI 访问顺序”两个要素类
"""

import arcpy
import math
import os
import random
import heapq


# ====================================================================
# Toolbox 类：工具箱入口，类名和结构固定
# ====================================================================
class Toolbox(object):
    def __init__(self):
        self.label = "遗传算法路径优化"
        self.alias = "GA_route"
        self.tools = [RouteOptimizationTool]


# ====================================================================
# RouteOptimizationTool 类：具体工具
# ====================================================================
class RouteOptimizationTool(object):

    def __init__(self):
        self.label = "最优路线"
        self.description = "基于遗传算法，在路网约束下求解旅游资源的最优访问顺序"
        self.canRunInBackground = False

    # ----------------------------------------------------------------
    # 参数定义
    # ----------------------------------------------------------------
    def getParameterInfo(self):

        p0 = arcpy.Parameter(
            displayName="输入旅游资源POI",
            name="input_points",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        p0.filter.list = ["Point", "MultiPoint"]

        p1 = arcpy.Parameter(
            displayName="编码字段（值必须唯一）",
            name="encode_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        p1.parameterDependencies = [p0.name]

        p2 = arcpy.Parameter(
            displayName="输入路网",
            name="input_lines",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        p2.filter.list = ["Polyline"]

        p3 = arcpy.Parameter(
            displayName="种群规模 N",
            name="pop_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        p3.value = 100

        p4 = arcpy.Parameter(
            displayName="多元锦标赛 k 值",
            name="tournament_k",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        p4.value = 3

        p5 = arcpy.Parameter(
            displayName="变异概率 m%（1~10）",
            name="mutation_rate",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input")
        p5.value = 5.0

        p6 = arcpy.Parameter(
            displayName="迭代次数 p",
            name="max_iterations",
            datatype="GPLong",
            parameterType="Required",
            direction="Input")
        p6.value = 200

        p7 = arcpy.Parameter(
            displayName="输出地理数据库（.gdb）",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        p8 = arcpy.Parameter(
            displayName="输出要素类前缀",
            name="output_prefix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        p8.value = "GA"

        p9 = arcpy.Parameter(
            displayName="启用 2-opt 后处理",
            name="use_2opt",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input")
        p9.value = True

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    # ----------------------------------------------------------------
    # 参数校验
    # ----------------------------------------------------------------
    def updateMessages(self, parameters):

        # 取值范围校验
        if parameters[3].value is not None:
            n = int(parameters[3].value)
            if n < 2:
                parameters[3].setErrorMessage("种群规模 N 至少为 2")

        if parameters[4].value is not None:
            k = int(parameters[4].value)
            if k < 2:
                parameters[4].setErrorMessage("锦标赛 k 值至少为 2")

        if parameters[3].value is not None and parameters[4].value is not None:
            n = int(parameters[3].value)
            k = int(parameters[4].value)
            if n < k:
                parameters[3].setErrorMessage("N 必须大于等于 k")
                parameters[4].setErrorMessage("k 必须小于等于 N")

        if parameters[5].value is not None:
            m = float(parameters[5].value)
            if m < 1 or m > 10:
                parameters[5].setErrorMessage("变异概率必须在 1% ~ 10% 之间")

        if parameters[6].value is not None:
            p = int(parameters[6].value)
            if p < 1:
                parameters[6].setErrorMessage("迭代次数必须大于 0")

        # 编码字段唯一性校验
        if parameters[0].value and parameters[1].value:
            try:
                pts = parameters[0].valueAsText
                field_name = parameters[1].valueAsText
                values = []
                has_null = False
                with arcpy.da.SearchCursor(pts, [field_name]) as cur:
                    for row in cur:
                        v = row[0]
                        if v is None:
                            has_null = True
                        values.append(v)
                if has_null:
                    parameters[1].setWarningMessage("编码字段存在空值，运行时会被自动编号")
                if len(values) != len(set(values)):
                    parameters[1].setErrorMessage("编码字段的值必须唯一")
                if len(values) < 2:
                    parameters[0].setErrorMessage("旅游资源POI数量至少为 2")
            except Exception as e:
                parameters[1].setErrorMessage("校验编码字段失败：{}".format(str(e)))

        # 输出数据库校验
        if parameters[7].value:
            gdb = parameters[7].valueAsText
            if not gdb.lower().endswith(".gdb"):
                parameters[7].setErrorMessage("请选择一个文件地理数据库（.gdb）作为输出位置")
            elif not arcpy.Exists(gdb):
                parameters[7].setErrorMessage("指定的地理数据库不存在")

        return

    # ----------------------------------------------------------------
    # 主执行逻辑
    # ----------------------------------------------------------------
    def execute(self, parameters, messages):

        arcpy.env.overwriteOutput = True

        # 读取参数
        input_points = parameters[0].valueAsText
        encode_field = parameters[1].valueAsText
        input_lines = parameters[2].valueAsText
        pop_size = int(parameters[3].value)
        tournament_k = int(parameters[4].value)
        mutation_rate = float(parameters[5].value) / 100.0
        max_iterations = int(parameters[6].value)
        output_gdb = parameters[7].valueAsText
        output_prefix = parameters[8].valueAsText or "GA"
        use_2opt = bool(parameters[9].value)

        messages.addMessage("=" * 50)
        messages.addMessage("参数读取完成")
        messages.addMessage("  旅游资源POI：{}".format(input_points))
        messages.addMessage("  编码字段：{}".format(encode_field))
        messages.addMessage("  路网：{}".format(input_lines))
        messages.addMessage("  N={}，k={}，m={}%，p={}".format(
            pop_size, tournament_k, int(mutation_rate * 100), max_iterations))
        messages.addMessage("  输出数据库：{}".format(output_gdb))
        messages.addMessage("=" * 50)

        # 坐标系检查
        self._check_spatial_reference(input_points, input_lines, messages)

        # 读取旅游资源POI
        tourism_raw = []
        auto_code = 0
        with arcpy.da.SearchCursor(input_points, ["SHAPE@XY", encode_field]) as cur:
            for row in cur:
                x, y = row[0]
                code = row[1]
                if code is None:
                    auto_code += 1
                    code = "P{}".format(auto_code)
                tourism_raw.append((round(x, 6), round(y, 6), code))

        if auto_code > 0:
            messages.addWarningMessage(
                "编码字段中有 {} 条空记录，已自动编号。".format(auto_code))

        n_genes = len(tourism_raw)
        if n_genes < 2:
            messages.addErrorMessage("旅游资源POI数量至少为 2，当前只有 {} 个。".format(n_genes))
            return

        messages.addMessage("读取到 {} 个旅游资源POI".format(n_genes))

        # 构建路网图
        graph = self._build_road_graph(input_lines)
        messages.addMessage("路网构建完成，节点数：{}".format(len(graph)))

        # 将旅游点吸附到路网
        tourism_nodes = self._snap_points_to_road(graph, tourism_raw, messages)
        messages.addMessage("旅游点吸附完成")

        # 连通路网各分量
        bridge_count = self._connect_components(graph)
        messages.addMessage("路网连通完成，添加桥接边：{}".format(bridge_count))

        # 计算距离矩阵
        dist_matrix, inf_pairs = self._build_distance_matrix(graph, tourism_nodes)
        if inf_pairs > 0:
            messages.addWarningMessage(
                "存在 {} 个不可达点对，尝试将孤立点连接到主路网...".format(inf_pairs))
            repaired = self._connect_isolated_to_road(graph, tourism_nodes)
            messages.addMessage("修复连接数：{}".format(repaired))
            dist_matrix, inf_pairs = self._build_distance_matrix(graph, tourism_nodes)

        if inf_pairs > 0:
            messages.addErrorMessage("修复后仍有 {} 个不可达点对，请检查路网数据。".format(inf_pairs))
            return

        messages.addMessage("距离矩阵计算完成")

        # 遗传算法主循环
        messages.addMessage("开始遗传算法优化...")
        best_chromo, best_fitness = self._genetic_algorithm(
            n_genes, pop_size, tournament_k, mutation_rate, max_iterations,
            dist_matrix, messages)

        # 2-opt 局部优化（后处理）
        if use_2opt:
            messages.addMessage("对最优解执行 2-opt 局部优化...")
            best_chromo = self._two_opt(best_chromo, dist_matrix)
            best_fitness = self._evaluate(best_chromo, dist_matrix)
            messages.addMessage("2-opt 完成，优化后路径长度：{:.2f}".format(best_fitness))

        coding = [str(tourism_raw[i][2]) for i in best_chromo]
        messages.addMessage("=" * 50)
        messages.addMessage("遗传算法完成")
        messages.addMessage("最优路径长度：{:.2f}".format(best_fitness))
        messages.addMessage("最优访问顺序：{}".format(" → ".join(coding)))
        messages.addMessage("=" * 50)

        # 输出结果
        self._write_results(
            input_points, output_gdb, output_prefix,
            tourism_raw, tourism_nodes, best_chromo, graph, messages)

        return

    # ====================================================================
    # 辅助方法
    # ====================================================================

    def _check_spatial_reference(self, input_points, input_lines, messages):
        """检查输入数据坐标系是否适合距离计算。"""
        try:
            for name, path in [("旅游资源POI", input_points), ("路网", input_lines)]:
                sr = arcpy.Describe(path).spatialReference
                if sr is None:
                    messages.addWarningMessage("{} 未定义坐标系。".format(name))
                elif sr.type == "Geographic":
                    messages.addWarningMessage(
                        "{} 使用地理坐标系（{}），距离单位为【度】，"
                        "建议先投影到以米为单位的投影坐标系。".format(name, sr.name))
                elif sr.type == "Projected":
                    messages.addMessage(
                        "{} 使用投影坐标系：{}，距离单位：{}".format(
                            name, sr.name, sr.linearUnitName or "未知"))
        except Exception as e:
            messages.addWarningMessage("坐标系检查失败：{}".format(str(e)))

    def _build_road_graph(self, input_lines):
        """将线要素转换为无向图邻接表。"""
        graph = {}
        with arcpy.da.SearchCursor(input_lines, ["SHAPE@"]) as cur:
            for row in cur:
                polyline = row[0]
                if polyline is None:
                    continue
                for part in polyline:
                    coords = []
                    for pt in part:
                        if pt is not None:
                            coords.append((round(pt.X, 6), round(pt.Y, 6)))
                    for i in range(len(coords) - 1):
                        a, b = coords[i], coords[i + 1]
                        length = math.hypot(a[0] - b[0], a[1] - b[1])
                        if length < 1e-10:
                            continue
                        graph.setdefault(a, {})[b] = length
                        graph.setdefault(b, {})[a] = length
        return graph

    def _snap_to_segment(self, px, py, ax, ay, bx, by):
        """将点投影到线段上，返回投影点坐标和距离。"""
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-14:
            return ax, ay, math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        sx = ax + t * dx
        sy = ay + t * dy
        return sx, sy, math.hypot(px - sx, py - sy)

    def _snap_points_to_road(self, graph, tourism_raw, messages):
        """将每个旅游点吸附到最近的路网边上，并返回吸附后的路网节点坐标列表。"""

        # 先收集所有原始线段，供投影使用
        line_segments = []
        visited_edges = set()
        for a in graph:
            for b in graph[a]:
                key = (a, b) if a < b else (b, a)
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                line_segments.append((a, b))

        # 为每个旅游点找最近线段和投影点
        snap_records = []
        for idx, (px, py, _) in enumerate(tourism_raw):
            best_dist = float('inf')
            best_snap = None
            best_ea = best_eb = None
            for a, b in line_segments:
                sx, sy, sd = self._snap_to_segment(px, py, a[0], a[1], b[0], b[1])
                if sd < best_dist:
                    best_dist = sd
                    best_snap = (round(sx, 6), round(sy, 6))
                    best_ea, best_eb = a, b
            snap_records.append((best_snap, best_ea, best_eb, idx))

        # 将投影点插入路网
        tourism_nodes = [None] * len(tourism_raw)
        inserted = 0
        skipped = 0

        for snap, ea, eb, idx in snap_records:
            # 情况 1：投影点本身已是路网节点
            if snap in graph:
                tourism_nodes[idx] = snap
                skipped += 1
                continue

            # 情况 2：投影点落在已知边 (ea, eb) 或 (eb, ea) 上，拆分该边
            found = False
            for start, end in [(ea, eb), (eb, ea)]:
                if start not in graph or end not in graph[start]:
                    continue
                sx, sy, sd = self._snap_to_segment(
                    snap[0], snap[1], start[0], start[1], end[0], end[1])
                if sd < 1e-6:
                    edge_len = graph[start][end]
                    len_a = math.hypot(snap[0] - start[0], snap[1] - start[1])
                    len_b = edge_len - len_a
                    # 删除旧边，添加新边
                    del graph[start][end]
                    graph.setdefault(snap, {})[start] = len_a
                    graph[start][snap] = len_a
                    graph.setdefault(snap, {})[end] = len_b
                    graph.setdefault(end, {})[snap] = len_b
                    tourism_nodes[idx] = snap
                    inserted += 1
                    found = True
                    break
            if found:
                continue

            # 情况 3：未找到对应边，直接连接到最近的两个路网节点
            nearest = []
            for node in graph:
                d = math.hypot(snap[0] - node[0], snap[1] - node[1])
                nearest.append((d, node))
            nearest.sort()
            for d, node in nearest[:2]:
                if d > 0:
                    graph.setdefault(snap, {})[node] = d
                    graph.setdefault(node, {})[snap] = d
            tourism_nodes[idx] = snap
            inserted += 1

        messages.addMessage("  吸附插入：{}，已位于节点：{}".format(inserted, skipped))
        return tourism_nodes

    def _connect_components(self, graph):
        """用最小生成树思路连接 graph 中的孤立连通分量。"""
        if not graph:
            return 0

        # BFS 找连通分量
        visited = set()
        comp_id = {}
        comp_idx = 0
        for node in list(graph.keys()):
            if node in visited:
                continue
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp_id[cur] = comp_idx
                for nb in graph.get(cur, {}):
                    if nb not in visited:
                        stack.append(nb)
            comp_idx += 1

        if comp_idx <= 1:
            return 0

        # 并查集
        parent = list(range(comp_idx))
        rank = [0] * comp_idx

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        # 收集分量间最短桥接边
        best_bridge = {}
        for na in graph:
            ca = comp_id.get(na, -1)
            for nb in graph[na]:
                cb = comp_id.get(nb, -1)
                if ca == cb or ca < 0 or cb < 0:
                    continue
                key = (min(ca, cb), max(ca, cb))
                d = graph[na][nb]
                if key not in best_bridge or d < best_bridge[key][0]:
                    best_bridge[key] = (d, na, nb)

        added = 0
        n_comp = comp_idx
        for d, na, nb in sorted(best_bridge.values(), key=lambda x: x[0]):
            ca, cb = comp_id[na], comp_id[nb]
            if union(ca, cb):
                graph.setdefault(na, {})[nb] = d
                graph.setdefault(nb, {})[na] = d
                added += 1
                n_comp -= 1
                if n_comp <= 1:
                    break
        return added

    def _connect_isolated_to_road(self, graph, tourism_nodes):
        """将不在主连通分量的旅游点连接到最近的主路网节点。"""
        # 找所有连通分量
        visited = set()
        components = []
        for node in graph:
            if node in visited:
                continue
            comp = set()
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for nb in graph.get(cur, {}):
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)

        if len(components) <= 1:
            return 0

        # 包含最多旅游点的主分量
        main_component = max(
            components,
            key=lambda c: sum(1 for n in tourism_nodes if n in c))

        repaired = 0
        for pt in tourism_nodes:
            if pt in main_component:
                continue
            best_dist = float('inf')
            best_node = None
            for node in main_component:
                d = math.hypot(pt[0] - node[0], pt[1] - node[1])
                if d < best_dist:
                    best_dist = d
                    best_node = node
            if best_node and best_dist > 0:
                graph.setdefault(pt, {})[best_node] = best_dist
                graph.setdefault(best_node, {})[pt] = best_dist
                repaired += 1
        return repaired

    def _dijkstra(self, graph, src):
        """Dijkstra 最短路径，返回 {节点: 距离}。"""
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float('inf')):
                continue
            for v, w in graph.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _dijkstra_path(self, graph, src, dst):
        """Dijkstra 最短路径，返回坐标列表；不可达返回 None。"""
        dist = {src: 0.0}
        prev = {src: None}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, float('inf')):
                continue
            for v, w in graph.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            return None
        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def _build_distance_matrix(self, graph, tourism_nodes):
        """计算旅游点两两之间的路网最短距离矩阵。"""
        n = len(tourism_nodes)
        matrix = [[float('inf')] * n for _ in range(n)]
        inf_pairs = 0
        for i in range(n):
            src = tourism_nodes[i]
            dists = self._dijkstra(graph, src)
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0.0
                    continue
                dst = tourism_nodes[j]
                d = dists.get(dst, float('inf'))
                matrix[i][j] = d
                if d == float('inf'):
                    inf_pairs += 1
        return matrix, inf_pairs

    def _evaluate(self, chromo, dist_matrix):
        """计算一条染色体（环线）的总路径长度。"""
        total = 0.0
        n = len(chromo)
        for i in range(n):
            a = chromo[i]
            b = chromo[(i + 1) % n]
            total += dist_matrix[a][b]
        return total

    def _two_opt(self, chromo, dist_matrix):
        """2-opt 局部搜索，直接修改并返回优化后的染色体。"""
        n = len(chromo)
        if n < 4:
            return chromo[:]

        improved = True
        while improved:
            improved = False
            for i in range(n):
                # j 至少与 i 间隔两个位置，避免重叠边
                for j in range(i + 2, n):
                    # 跳过 i=0 且 j=n-1 的情况，避免与边 (n-1,0) 重复
                    if i == 0 and j == n - 1:
                        continue
                    a = chromo[i]
                    b = chromo[(i + 1) % n]
                    c = chromo[j]
                    d = chromo[(j + 1) % n]

                    delta = (dist_matrix[a][c] + dist_matrix[b][d]
                             - dist_matrix[a][b] - dist_matrix[c][d])

                    if delta < -1e-12:
                        # 反转 i+1 到 j 之间的片段
                        chromo[i + 1:j + 1] = reversed(chromo[i + 1:j + 1])
                        improved = True
        return chromo

    def _tournament_select(self, population, fitness_list, k):
        """k 元锦标赛选择，返回适应度最优的个体副本。"""
        indices = random.sample(range(len(population)), k)
        best_idx = min(indices, key=lambda i: fitness_list[i])
        return population[best_idx][:]

    def _order_crossover(self, a, b):
        """顺序交叉（Order Crossover, OX）。"""
        n = len(a)
        i, j = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[i:j + 1] = a[i:j + 1]
        fixed = set(child[i:j + 1])

        # 从 b 的第二个交叉点后循环读取，剔除已存在基因
        remaining = []
        pos = (j + 1) % n
        for _ in range(n):
            if b[pos] not in fixed:
                remaining.append(b[pos])
            pos = (pos + 1) % n

        # 填充 child 的空缺
        fill_pos = (j + 1) % n
        for gene in remaining:
            while child[fill_pos] is not None:
                fill_pos = (fill_pos + 1) % n
            child[fill_pos] = gene
            fill_pos = (fill_pos + 1) % n
        return child

    def _swap_mutation(self, chromo):
        """交换变异：随机交换两个基因位置。"""
        n = len(chromo)
        i, j = random.sample(range(n), 2)
        chromo[i], chromo[j] = chromo[j], chromo[i]

    def _genetic_algorithm(self, n_genes, pop_size, tournament_k, mutation_rate,
                           max_iterations, dist_matrix, messages):
        """遗传算法主循环，返回最优染色体和最优适应度。"""

        # 初始化种群
        base = list(range(n_genes))
        population = []
        for _ in range(pop_size):
            chromo = base[:]
            random.shuffle(chromo)
            population.append(chromo)

        fitness_list = [self._evaluate(c, dist_matrix) for c in population]

        best_idx = min(range(pop_size), key=lambda i: fitness_list[i])
        best_fitness = fitness_list[best_idx]
        best_chromo = population[best_idx][:]

        for gen in range(max_iterations):
            children = []
            children_fitness = []

            for _ in range(pop_size):
                a = self._tournament_select(population, fitness_list, tournament_k)
                b = self._tournament_select(population, fitness_list, tournament_k)
                child = self._order_crossover(a, b)
                if random.random() < mutation_rate:
                    self._swap_mutation(child)
                children.append(child)
                children_fitness.append(self._evaluate(child, dist_matrix))

            # (μ + λ) 精英保留
            combined = population + children
            combined_fitness = fitness_list + children_fitness
            ranked = sorted(range(len(combined)), key=lambda i: combined_fitness[i])

            population = [combined[i][:] for i in ranked[:pop_size]]
            fitness_list = [combined_fitness[i] for i in ranked[:pop_size]]

            if fitness_list[0] < best_fitness:
                best_fitness = fitness_list[0]
                best_chromo = population[0][:]

            if gen % 100 == 0 or gen == max_iterations - 1:
                messages.addMessage(
                    "  迭代 {}：当代最优={:.2f}，全局最优={:.2f}".format(
                        gen, fitness_list[0], best_fitness))

        return best_chromo, best_fitness

    def _write_results(self, input_points, output_gdb, output_prefix,
                       tourism_raw, tourism_nodes, best_chromo, graph, messages):
        """输出最优路径折线和 POI 访问顺序。"""

        route_name = arcpy.ValidateTableName("{}_Route".format(output_prefix), output_gdb)
        poi_name = arcpy.ValidateTableName("{}_POI_Order".format(output_prefix), output_gdb)
        route_path = os.path.join(output_gdb, route_name)
        poi_path = os.path.join(output_gdb, poi_name)

        if arcpy.Exists(route_path):
            arcpy.management.Delete(route_path)
        if arcpy.Exists(poi_path):
            arcpy.management.Delete(poi_path)

        sr = arcpy.Describe(input_points).spatialReference

        # 1. 输出路径折线
        arcpy.management.CreateFeatureclass(
            output_gdb, route_name, "POLYLINE", spatial_reference=sr)
        arcpy.management.AddField(route_path, "PathOrder", "LONG")

        with arcpy.da.InsertCursor(route_path, ["SHAPE@", "PathOrder"]) as cursor:
            n = len(best_chromo)
            for i in range(n):
                a_idx = best_chromo[i]
                b_idx = best_chromo[(i + 1) % n]
                src = tourism_nodes[a_idx]
                dst = tourism_nodes[b_idx]
                path_coords = self._dijkstra_path(graph, src, dst)
                if path_coords:
                    arr = arcpy.Array([arcpy.Point(x, y) for x, y in path_coords])
                    cursor.insertRow([arcpy.Polyline(arr), i])

        messages.addMessage("最优路径已保存：{}".format(route_path))

        # 2. 输出 POI 访问顺序
        visit_order = {gene_idx: order for order, gene_idx in enumerate(best_chromo)}
        arcpy.management.CopyFeatures(input_points, poi_path)
        arcpy.management.AddField(poi_path, "VisitOrder", "LONG")

        idx = 0
        with arcpy.da.UpdateCursor(poi_path, ["VisitOrder"]) as cursor:
            for row in cursor:
                row[0] = visit_order.get(idx, -1)
                cursor.updateRow(row)
                idx += 1

        messages.addMessage("POI 访问顺序已保存：{}".format(poi_path))

        # 尝试添加到当前地图
        try:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            m = aprx.activeMap
            if m:
                m.addDataFromPath(route_path)
                m.addDataFromPath(poi_path)
                messages.addMessage("结果已添加到当前地图。")
        except Exception:
            pass

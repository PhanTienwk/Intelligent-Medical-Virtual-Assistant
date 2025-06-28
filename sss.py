from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, collect_list, explode, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark import SparkContext, StorageLevel
import sys
from typing import List, Tuple, Dict


class PersonalizedPageRank:
    def __init__(self, app_name="PersonalizedPageRank"):
        """Khởi tạo Spark Session"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        self.sc = self.spark.sparkContext
        self.sc.setLogLevel("WARN")

    def compute_personalized_pagerank(
            self,
            edges: List[Tuple[str, str]],
            source_page: str,
            damping_factor: float = 0.85,
            max_iterations: int = 100,
            tolerance: float = 1e-6
    ) -> List[Tuple[str, float]]:
        """
        Tính Personalized PageRank sử dụng RDD

        Args:
            edges: Danh sách các cạnh [(from, to), ...]
            source_page: Trang nguồn
            damping_factor: Hệ số damping (thường là 0.85)
            max_iterations: Số iteration tối đa
            tolerance: Ngưỡng hội tụ

        Returns:
            List[(page_id, pagerank_value)]
        """

        print(f"🚀 Bắt đầu tính Personalized PageRank với trang nguồn: {source_page}")
        print(f"📊 Tổng số cạnh: {len(edges)}")

        # Bước 1: Tạo RDD từ danh sách edges
        edges_rdd = self.sc.parallelize(edges)

        # Bước 2: Xây dựng cấu trúc đồ thị - adjacency list
        # Tạo outlinks: page_id -> [list_of_outgoing_pages]
        outlinks = edges_rdd \
            .groupByKey() \
            .mapValues(list) \
            .persist(StorageLevel.MEMORY_AND_DISK)

        # Lấy tất cả các trang duy nhất
        all_pages = edges_rdd.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
        all_pages_list = all_pages.collect()

        print(f"📝 Tổng số trang: {len(all_pages_list)}")

        # Bước 3: Khởi tạo PageRank
        # Trang nguồn = 1.0, các trang khác = 0.0
        def initialize_rank(page_id):
            return 1.0 if page_id == source_page else 0.0

        page_ranks = all_pages.map(lambda page: (page, initialize_rank(page))) \
            .persist(StorageLevel.MEMORY_AND_DISK)

        print("✅ Khởi tạo PageRank hoàn thành")

        # Bước 4: Tính toán lặp
        iteration = 0
        converged = False
        previous_ranks = None

        while iteration < max_iterations and not converged:
            iteration += 1

            # Tính contributions từ mỗi trang đến các trang đích
            contributions = page_ranks \
                .join(outlinks) \
                .flatMap(lambda x: self._calculate_contributions(x[0], x[1][0], x[1][1]))

            # Tổng hợp contributions cho mỗi trang
            aggregated_contributions = contributions.reduceByKey(lambda a, b: a + b)

            # Tính PageRank mới
            new_page_ranks = all_pages.map(lambda page: (page, 0.0)) \
                .leftOuterJoin(aggregated_contributions) \
                .map(lambda x: self._calculate_new_rank(
                x[0], x[1][1], source_page, damping_factor
            )) \
                .persist(StorageLevel.MEMORY_AND_DISK)

            # Kiểm tra hội tụ
            if iteration > 1:
                rank_differences = previous_ranks.join(new_page_ranks) \
                    .map(lambda x: abs(x[1][1] - x[1][0]))

                max_difference = rank_differences.max()
                converged = max_difference < tolerance

                print(f"🔄 Iteration {iteration}: Max difference = {max_difference:.8f}")

            # Cập nhật cho iteration tiếp theo
            if previous_ranks:
                previous_ranks.unpersist()
            previous_ranks = page_ranks
            page_ranks = new_page_ranks

        # Kết thúc
        if converged:
            print(f"✅ Hội tụ sau {iteration} iterations")
        else:
            print(f"⚠️  Đạt số iteration tối đa ({max_iterations})")

        # Lấy kết quả
        results = page_ranks.collect()

        # Dọn dẹp
        outlinks.unpersist()
        if previous_ranks:
            previous_ranks.unpersist()
        page_ranks.unpersist()

        return results

    def _calculate_contributions(self, page_id: str, rank: float, outgoing_pages: List[str]):
        """Tính contribution từ một trang đến các trang đích"""
        if not outgoing_pages:
            return []

        contribution_per_page = rank / len(outgoing_pages)
        return [(target_page, contribution_per_page) for target_page in outgoing_pages]

    def _calculate_new_rank(self, page_id: str, contribution_opt, source_page: str, damping_factor: float):
        """Tính PageRank mới cho một trang"""
        contribution = contribution_opt if contribution_opt is not None else 0.0

        if page_id == source_page:
            # Trang nguồn: damped contribution + jump probability
            new_rank = damping_factor * contribution + (1.0 - damping_factor)
        else:
            # Trang khác: chỉ có damped contribution
            new_rank = damping_factor * contribution

        return (page_id, new_rank)

    def compute_with_dataframes(
            self,
            edges: List[Tuple[str, str]],
            source_page: str,
            damping_factor: float = 0.85,
            max_iterations: int = 100,
            tolerance: float = 1e-6
    ) -> List[Tuple[str, float]]:
        """
        Tính Personalized PageRank sử dụng DataFrame (cách tiếp cận hiện đại hơn)
        """

        print(f"🚀 Tính PageRank bằng DataFrame với trang nguồn: {source_page}")

        # Tạo DataFrame từ edges
        edges_df = self.spark.createDataFrame(edges, ["from_page", "to_page"])

        # Lấy tất cả trang duy nhất
        all_pages_df = edges_df.select(col("from_page").alias("page_id")) \
            .union(edges_df.select(col("to_page").alias("page_id"))) \
            .distinct()

        # Khởi tạo ranks
        initial_ranks_df = all_pages_df.withColumn(
            "rank",
            when(col("page_id") == source_page, 1.0).otherwise(0.0)
        )

        # Tạo adjacency list
        outlinks_df = edges_df.groupBy("from_page") \
            .agg(collect_list("to_page").alias("outlinks"))

        current_ranks_df = initial_ranks_df

        for iteration in range(1, max_iterations + 1):
            print(f"🔄 DataFrame Iteration {iteration}")

            # Tính contributions (phức tạp hơn với DataFrame)
            # Để đơn giản, chúng ta convert về RDD cho phần này
            current_ranks_rdd = current_ranks_df.rdd.map(lambda row: (row.page_id, row.rank))
            outlinks_rdd = outlinks_df.rdd.map(lambda row: (row.from_page, row.outlinks))

            # ... (logic tương tự như RDD version)
            break  # Placeholder - implementation đầy đủ sẽ phức tạp hơn

        return current_ranks_df.collect()

    def load_edges_from_file(self, file_path: str) -> List[Tuple[str, str]]:
        """Đọc edges từ file text"""
        try:
            edges_rdd = self.sc.textFile(file_path) \
                .map(lambda line: line.strip().split()) \
                .filter(lambda parts: len(parts) >= 2) \
                .map(lambda parts: (parts[0], parts[1]))

            return edges_rdd.collect()
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {e}")
            return []

    def save_results(self, results: List[Tuple[str, float]], output_path: str):
        """Lưu kết quả ra file"""
        try:
            results_rdd = self.sc.parallelize(results) \
                .map(lambda x: f"{x[0]}\t{x[1]:.6f}")

            results_rdd.saveAsTextFile(output_path)
            print(f"✅ Đã lưu kết quả tại: {output_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file: {e}")

    def validate_results(self, results: List[Tuple[str, float]]):
        """Kiểm tra tính hợp lệ của kết quả"""
        total_rank = sum(rank for _, rank in results)
        print(f"📊 Tổng PageRank: {total_rank:.6f}")

        # Trong Personalized PageRank, tổng rank sẽ khác với PageRank thông thường
        print(f"📈 Các trang có rank cao nhất:")
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        for i, (page_id, rank) in enumerate(sorted_results[:5]):
            print(f"   {i + 1}. {page_id}: {rank:.6f}")

    def stop(self):
        """Dừng Spark Session"""
        self.spark.stop()


def main():
    """Hàm main để test"""

    # Khởi tạo PersonalizedPageRank
    ppr = PersonalizedPageRank()

    try:
        # Ví dụ đồ thị đơn giản: A -> B -> C -> A
        print("=" * 50)
        print("🧪 TEST VỚI ĐỒ THỊ ĐỚN GIẢN")
        print("=" * 50)

        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("A", "C"),  # Thêm cạnh để phức tạp hơn
            ("B", "A")
        ]

        # Tính Personalized PageRank với A là source
        print("\n🎯 Tính với trang nguồn = A")
        results_A = ppr.compute_personalized_pagerank(
            edges=edges,
            source_page="A",
            damping_factor=0.85,
            max_iterations=20,
            tolerance=1e-6
        )

        print("\n📊 KẾT QUẢ VỚI SOURCE = A:")
        ppr.validate_results(results_A)

        print("\n" + "=" * 30)

        # Tính với B là source để so sánh
        print("\n🎯 Tính với trang nguồn = B")
        results_B = ppr.compute_personalized_pagerank(
            edges=edges,
            source_page="B",
            damping_factor=0.85,
            max_iterations=20,
            tolerance=1e-6
        )

        print("\n📊 KẾT QUẢ VỚI SOURCE = B:")
        ppr.validate_results(results_B)

        # So sánh kết quả
        print("\n🔍 SO SÁNH KẾT QUẢ:")
        print("Source A vs Source B:")
        results_A_dict = dict(results_A)
        results_B_dict = dict(results_B)

        for page in ["A", "B", "C"]:
            rank_A = results_A_dict.get(page, 0)
            rank_B = results_B_dict.get(page, 0)
            print(f"  Trang {page}: Source=A ({rank_A:.4f}) vs Source=B ({rank_B:.4f})")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Dừng Spark
        ppr.stop()


def example_with_larger_graph():
    """Ví dụ với đồ thị lớn hơn"""

    ppr = PersonalizedPageRank("LargerGraph-PageRank")

    try:
        # Đồ thị phức tạp hơn
        edges = [
            ("A", "B"), ("A", "C"), ("A", "D"),
            ("B", "C"), ("B", "E"),
            ("C", "D"), ("C", "F"),
            ("D", "A"), ("D", "F"),
            ("E", "B"), ("E", "F"),
            ("F", "A"), ("F", "C")
        ]

        print("🏗️  ĐỒ THỊ PHỨC TẠP HỚN")
        print(f"Số cạnh: {len(edges)}")
        print(f"Các cạnh: {edges}")

        # Tính với nhiều source khác nhau
        for source in ["A", "D", "F"]:
            print(f"\n🎯 Source = {source}")
            results = ppr.compute_personalized_pagerank(
                edges=edges,
                source_page=source,
                damping_factor=0.85,
                max_iterations=50
            )

            print(f"📊 Top 3 trang quan trọng nhất từ {source}:")
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            for i, (page_id, rank) in enumerate(sorted_results[:3]):
                print(f"   {i + 1}. {page_id}: {rank:.6f}")

    finally:
        ppr.stop()


if __name__ == "__main__":
    print("🐍 PERSONALIZED PAGERANK VỚI PYSPARK")
    print("=" * 60)

    # Chạy test cơ bản
    main()

    print("\n" + "=" * 60)
    print("🔍 TEST VỚI ĐỒ THỊ LỚN HỚN")
    print("=" * 60)

    # Chạy test với đồ thị phức tạp
    example_with_larger_graph()

    print("\n✅ HOÀN THÀNH TẤT CẢ TEST!")
#include "types/Scalar.h"

#include <gtest/gtest.h>

#ifdef SSE2

TEST(ScalarSIMDTest, Assignment)
{
    ScalarSIMD a(1.0, 2.0);

    EXPECT_DOUBLE_EQ(a.asArray()[0], 1.0);
    EXPECT_DOUBLE_EQ(a.asArray()[1], 2.0);
}

#endif